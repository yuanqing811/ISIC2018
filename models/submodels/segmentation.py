from models.layers import SubPixelUpscaling
from models.layers import UpsampleLike
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.utils import conv_utils
from misc_utils.model_utils import name_or_none

from initializers import PriorProbability
from keras.initializers import Zeros


def __conv_block(filters,
                 activation='relu',
                 kernel_size=3,
                 strides=1,
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 block_prefix=None):

    def block(x):
        for i in range(len(filters)):
            x = Conv2D(filters[i],
                       conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size'),
                       strides=strides,
                       dilation_rate=conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate'),
                       activation=None,
                       padding='same',
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       name=name_or_none(block_prefix, '_conv{a}-k{b}-d{c}'.format(a=(i+1),
                                                                                   b=kernel_size,
                                                                                   c=dilation_rate)))(x)

            if activation.lower() == 'leakyrelu':
                x = LeakyReLU(alpha=0.33)(x)
            else:
                x = Activation(activation)(x)
        return x
    return block


def __transition_up_block(filters,
                          merge_size,
                          upsampling_type='deconv',
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          block_prefix=None):
    """Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsample', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.

    # Returns
        a keras tensor
    """
    if upsampling_type not in {'upsample', 'subpixel', 'deconv', 'resize'}:
        raise ValueError('upsampling_type must be in  {`upsample`, `subpixel`, `deconv`, `resize`}: %s' % str(upsampling_type))

    merge_size = conv_utils.normalize_tuple(merge_size, 2, 'merge_size')

    def block(ip):
        try:
            src, dst = ip
        except TypeError:
            src = ip
            dst = None

        # copy and crop
        if K.image_data_format() == 'channels_last':
            indices = slice(1, 3)
            channel_axis = -1
        else:
            indices = slice(2, 4)
            channel_axis = 1

        src_height, src_width = K.get_variable_shape(src)[indices]

        target_height, target_width = merge_size
        scale_factor = ((target_height + src_height - 1) // src_height,
                        (target_width + src_width - 1) // src_width)

        # upsample and crop
        if upsampling_type == 'upsample':
            x = UpsampleLike(scale_factor=scale_factor,
                             name=name_or_none(block_prefix, '_upsampling'))(src)
            x = Conv2D(filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       name=name_or_none(block_prefix, '_conv'))(x)
        elif upsampling_type == 'subpixel':
            x = Conv2D(filters,
                       kernel_size=(2, 2),
                       padding='same',
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       name=name_or_none(block_prefix, '_conv'))(src)
            x = SubPixelUpscaling(scale_factor=scale_factor,
                                  name=name_or_none(block_prefix, '_subpixel'))(x)
        else:
            x = Conv2DTranspose(filters,
                                kernel_size=(scale_factor * 2, scale_factor * 2),
                                strides=scale_factor,
                                padding='same',
                                kernel_initializer=kernel_initializer,
                                use_bias=False,
                                # bias_initializer=bias_initializer,
                                name=name_or_none(block_prefix, '_deconv'))(src)

        if src_height * scale_factor[0] > target_height or src_width * scale_factor[1] > target_width:
            height_padding, width_padding = (src_height - target_height) // 2, (src_width - target_width) // 2
            x = Cropping2D(cropping=(height_padding, width_padding),
                           name=name_or_none(block_prefix, 'crop1'))(x)

        if dst is None:
            return x

        dst_height, dst_width = K.get_variable_shape(dst)[indices]

        # copy and crop
        if dst_height > target_height or dst_width > target_width:
            height_padding, width_padding = ((dst_height - target_height) // 2, (dst_width - target_width) // 2)
            dst = Cropping2D(cropping=(height_padding, width_padding),
                             name=name_or_none(block_prefix, 'crop2'))(dst)

        x = Concatenate(axis=channel_axis, name=name_or_none(block_prefix, '_merge'))([x, dst])

        return x

    return block


# todo: can be made more efficient
def __normalize_target_size(curr_size, target_size, scale_factor):
    while curr_size < target_size:
        target_size //= scale_factor
    return target_size


def default_decoder_model(features,
                          num_classes,
                          output_size,
                          scale_factor,
                          blocks,
                          layers_per_block,
                          upsampling_type='deconv',
                          activation='relu',
                          dilation_rate=1,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          prior_probability=0.01,
                          bottleneck=False,
                          use_activation=True,
                          include_top=True):
    """
    :param features:            list of features from encoder
    :param output_size:         size of the output segmentation mask
    :param num_classes:         The number of classes of pixels.
    :param blocks:              The number of filters for each block of the decoder
    :param dilation_rate:       Dilation rate of the conv layers for each block of the decoder
    :param layers_per_block:    Number of layers for each conv block.
    :param scale_factor:        The rate at which the size grows
    :param upsampling_type:     Upsampling type
    :param activation:          activation of conv blocks
    :param kernel_initializer:  Conv2D kernel initializer. default: 'glorot_uniform',
    :param bias_initializer:    Conv2D bias initializer. default: 'zeros',
    :param prior_probability:   initializer for output conv layer
    :param use_activation:      whether to use activation at the output layer
    :param include_top:         whether to include the top layer
    :param bottleneck:          add bottleneck at the output of encoder
    :return:                    A keras.model.Model that predicts classes
    """

    output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
    output_height, output_width = output_size

    indices = slice(1, 3) if K.image_data_format() == 'channels_last' else slice(2, 4)
    channel = 3 if K.image_data_format() == 'channels_last' else 1

    feature_shapes = [K.get_variable_shape(feature) for feature in features]
    feature_sizes = [feature_shape[indices] for feature_shape in feature_shapes]

    feature_height, feature_width = feature_sizes[0]
    if feature_height < output_height or feature_width < output_width:
        features.insert(0, None)
        feature_shapes.insert(0, None)
        feature_sizes.insert(0, output_size)

    num_features = len(features)
    num_blocks = len(blocks)

    assert num_features == num_blocks, 'num_features != num_blocks'

    dilation_rate = conv_utils.normalize_tuple(dilation_rate, num_blocks, 'dilation_rate')

    if bottleneck:
        for i in range(num_blocks-1, -1, -1):
            if features[i] is not None and feature_shapes[i][channel] > blocks[i]:
                features[i] = Conv2D(blocks[i], (1, 1),
                                     activation='relu',
                                     padding='same',
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='feature%d_bottleneck' % (i+1))(features[i])

    layers_per_block = conv_utils.normalize_tuple(layers_per_block, num_features, 'layers_per_block')

    x = features[-1]

    num_blocks = len(blocks)
    for i in range(num_blocks-1, 0, -1):
        dst = features[i-1]
        dst_height, dst_width = feature_sizes[i-1]
        merge_size = __normalize_target_size(dst_height, output_height, scale_factor)

        if dst_width != dst_height:
            merge_size = (merge_size, __normalize_target_size(dst_width, output_width, scale_factor))

        x = __transition_up_block(filters=blocks[i - 1],
                                  merge_size=merge_size,
                                  upsampling_type=upsampling_type,
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer,
                                  block_prefix='feature%d' % (i+1)
                                  )([x, dst])

        if layers_per_block[i-1] > 0:
            x = __conv_block(filters=conv_utils.normalize_tuple(blocks[i-1],
                                                                layers_per_block[i-1],
                                                                'filters'),
                             activation=activation,
                             dilation_rate=dilation_rate[i-1],
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             block_prefix='feature%d' % i)(x)

    if include_top:
        x = Conv2D(num_classes, (1, 1),
                   padding='same',
                   kernel_initializer=Zeros(),
                   bias_initializer=PriorProbability(probability=prior_probability),
                   name='predictions')(x)
        if use_activation:
            output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
            x = Activation(output_activation, name='outputs')(x)

    return x



