import keras

# from models.layers import SubPixelUpscaling
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import Conv2DTranspose
from keras.utils import conv_utils
from misc_utils.model_utils import name_or_none


def __conv_block(nb_filters,
                 activation='relu',
                 block_prefix=None):

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        # 'kernel_initializer': kernel_initializer,
        # 'bias_initializer': bias_initializer,
    }

    nb_layers_per_block = 1 if isinstance(nb_filters, int) else len(nb_filters)
    nb_filters = conv_utils.normalize_tuple(nb_filters, nb_layers_per_block, 'nb_filters')

    def block(x):
        for i, n in enumerate(nb_filters):
            x = Conv2D(filters=nb_filters[i],
                       name=name_or_none(block_prefix, '_conv%d' % (i+1)),
                       **options)(x)

            if activation.lower() == 'leakyrelu':
                x = LeakyReLU(alpha=0.33)(x)
            else:
                x = Activation(activation)(x)
        return x
    return block


def __transition_up_block(nb_filters,
                          merge_size,
                          upsampling_type='deconv',
                          block_prefix=None):
    """Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
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
    options = {
        'padding': 'same'
    }

    if upsampling_type not in {'upsample', 'subpixel', 'deconv'}:
        raise ValueError('upsampling_type must be in  {`upsample`, `subpixel`, `deconv`}: %s' % str(upsampling_type))

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
            x = UpSampling2D(size=scale_factor,
                             name=name_or_none(block_prefix, '_upsampling'))(src)
            x = Conv2D(nb_filters, (2, 2),
                       activation='relu', padding='same', name=name_or_none(block_prefix, '_conv'))(x)
        elif upsampling_type == 'subpixel':
            x = Conv2D(nb_filters, (2, 2),
                       activation='relu', padding='same', name=name_or_none(block_prefix, '_conv'))(src)
            x = SubPixelUpscaling(scale_factor=scale_factor,
                                  name=name_or_none(block_prefix, '_subpixel'))(x)
        else:
            x = Conv2DTranspose(nb_filters, (2, 2), strides=scale_factor,
                                name=name_or_none(block_prefix, '_deconv'),
                                **options)(src)

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
                          num_classes=1,
                          output_size=224,
                          scale_factor=2,
                          init_nb_filters=64,
                          growth_rate=2,
                          nb_layers_per_block=2,
                          max_nb_filters=512,
                          upsampling_type='deconv',
                          activation='relu',
                          bottleneck=False,
                          use_activation=True,
                          include_top=True):
    """
    :param features:            list of features from encoder
    :param output_size:         size of the output segmentation mask
    :param num_classes:         The number of classes of pixels.
    :param init_nb_filters:     Number of filters for last conv block.
    :param growth_rate:         The rate at which the number of filters grow from block to block
    :param nb_layers_per_block: Number of layers for each conv block.
    :param max_nb_filters:      max # of filters
    :param scale_factor:        The rate at which the size grows
    :param upsampling_type:     Upsampling type
    :param activation:          activation of conv blocks
    :param use_activation:      whether to use activation of output layer
    :param include_top:         whether to use the top layer
    :param bottleneck:          add bottleneck at the output of encoder
    :return: A keras.model.Model that predicts classes
    """

    output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
    output_height, output_width = output_size

    __init_nb_filters = init_nb_filters
    indices = slice(1, 3) if K.image_data_format() == 'channels_last' else slice(2, 4)
    channel = 3 if K.image_data_format() == 'channels_last' else 1

    nb_features = len(features)
    feature_shapes = [K.get_variable_shape(feature) for feature in features]
    feature_sizes = [feature_shape[indices] for feature_shape in feature_shapes]

    feature_height, feature_width = feature_sizes[0]
    if feature_height < output_height or feature_width < output_width:
        __init_nb_filters = int(__init_nb_filters * growth_rate)

    if bottleneck:
        for i in range(nb_features - 1, -1, -1):
            feature_shape = feature_shapes[i]
            nb_filters = int(__init_nb_filters * (growth_rate ** i))
            nb_filters = min(nb_filters, max_nb_filters)
            if feature_shape[channel] > nb_filters:
                features[i] = Conv2D(nb_filters, 1,
                                     padding='same',
                                     activation='relu',
                                     name='feature%d_bottleneck' % (i+1))(features[i])

    nb_layers_per_block = conv_utils.normalize_tuple(nb_layers_per_block, nb_features, 'nb_layers_per_block')

    x = features[-1]

    for i in range(nb_features-1, 0, -1):
        dst = features[i-1]
        dst_height, dst_width = feature_sizes[i-1]

        merge_size = __normalize_target_size(dst_height, output_height, scale_factor)
        if dst_width != dst_height:
            merge_size = (merge_size, __normalize_target_size(dst_width, output_width, scale_factor))

        nb_filters = int(__init_nb_filters * (growth_rate ** (i-1)))
        nb_filters = min(nb_filters, max_nb_filters)

        x = __transition_up_block(nb_filters=nb_filters,
                                  merge_size=merge_size,
                                  block_prefix='feature%d' % (i+1),
                                  upsampling_type=upsampling_type)([x, dst])

        x = __conv_block(nb_filters=conv_utils.normalize_tuple(nb_filters,
                                                               nb_layers_per_block[i-1],
                                                               'nb_filters'),
                         activation=activation,
                         block_prefix='feature%d' % i)(x)

    if __init_nb_filters > init_nb_filters:
        x = __transition_up_block(nb_filters=init_nb_filters,
                                  merge_size=output_size,
                                  block_prefix='decoder_block%d' % (nb_features+1),
                                  upsampling_type=upsampling_type)(x)

        x = __conv_block(nb_filters=[init_nb_filters],
                         activation=activation,
                         block_prefix='feature%d' %(nb_features + 1))(x)

    if include_top:
        x = Conv2D(num_classes, (1, 1), activation='linear', name='predictions')(x)
        if use_activation:
            output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
            x = Activation(output_activation, name='outputs')(x)

    return x



