from keras import backend as K
from keras.initializers import Zeros, RandomNormal
from keras.layers import Concatenate, Conv2D, Activation, Cropping2D, Add, Conv2DTranspose
from keras.utils import conv_utils

from initializers import PriorProbability
from models.layers import UpsampleLike
from misc_utils.model_utils import name_or_none


def normalize_target_size(image_size, scale_factor, target_size):
    while image_size < target_size:
        target_size //= scale_factor
    return target_size


def decoder2(features,
             num_classes,
             output_size,
             scale_factor,
             filters=256,
             prior_probability=0.01,
             use_activation=True):

    output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')

    indices = slice(1, 3) if K.image_data_format() == 'channels_last' else slice(2, 4)
    channel = 3 if K.image_data_format() == 'channels_last' else 1

    resized = []

    for i, x in enumerate(features):
        feature_shape = K.get_variable_shape(x)
        feature_size = feature_shape[indices]
        x = Conv2D(filters, (1, 1), padding='same', name='reduced%d' % (i+1))(x)

        if feature_size[0] < output_size[0] or feature_size[1] < output_size[1]:
            x = UpsampleLike(scale_factor=scale_factor,
                             target_size=output_size,
                             name='upsampled%d' % (i+1))(x)

        x = Conv2D(filters, (3, 3), padding='same', name='feature%d' % (i+1))(x)

        resized.append(x)

    x = Concatenate(axis=channel, name='merged')(resized)

    for i in range(4):
        x = Conv2D(filters, kernel_size=3,
                   strides=1, padding='same',
                   activation='relu',
                   name='merged%d' % (i+1))(x)

    x = Conv2D(num_classes, (1, 1),
               padding='same',
               kernel_initializer=Zeros(),
               bias_initializer=PriorProbability(probability=prior_probability),
               name='score')(x)

    if use_activation:
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        x = Activation(output_activation, name='outputs')(x)

    return x


def __transition_up_block(filters,
                          target_size,
                          upsampling_type='deconv',
                          merge_type='concatenate',
                          kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                          bias_initializer='zeros',
                          block_prefix=None):
    """Adds an upsampling block. Upsampling op=eration relies on the the type parameter.

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
    if upsampling_type not in {'upsample', 'deconv'}:
        raise ValueError('upsampling_type must be in  {`upsample`, `deconv`}: %s' % str(upsampling_type))

    merge_size = conv_utils.normalize_tuple(target_size, 2, 'merge_size')

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
            x = Conv2D(filters, kernel_size=3,
                       strides=1, padding='same',
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       name=name_or_none(block_prefix, '_conv'))(x)
        else:
            x = Conv2DTranspose(
                filters,
                (scale_factor[0] * 2, scale_factor[1] * 2),
                strides=scale_factor,
                padding='same',
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                # use_bias=False,
                name=name_or_none(block_prefix, '_deconv')
            )(src)

        if src_height * scale_factor[0] > target_height or src_width * scale_factor[1] > target_width:
            cropping = (src_height - target_height) // 2, (src_width - target_width) // 2
            x = Cropping2D(cropping=cropping,
                           name=name_or_none(block_prefix, 'crop1'))(x)

        if dst is None:
            return x

        dst_height, dst_width = K.get_variable_shape(dst)[indices]

        # copy and crop
        if dst_height > target_height or dst_width > target_width:
            cropping = ((dst_height - target_height) // 2, (dst_width - target_width) // 2)
            dst = Cropping2D(cropping=cropping,
                             name=name_or_none(block_prefix, 'crop2'))(dst)

        if merge_type is 'concatenate':
            x = Concatenate(axis=channel_axis, name=name_or_none(block_prefix, '_merge'))([x, dst])
        else:
            x = Add()([x, dst])

        return x

    return block


def decoder3(features,
             num_classes,
             output_size,
             scale_factor,
             use_activation=True):

    # fully-connected layers converted to convolutional layers
    output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
    indices = slice(1, 3) if K.image_data_format() == 'channels_last' else slice(2, 4)

    num_features = len(features)
    features = [Conv2D(num_classes, (1, 1),
                       padding='same',
                       kernel_initializer='zeros',
                       bias_initializer=PriorProbability(probability=0.01),
                       name='score%d' % (i+1))(x)
                for i, x in enumerate(features)]

    feature_sizes = [K.get_variable_shape(feature)[indices] for feature in features]

    x = features[-1]

    for i in range(num_features-1, 0, -1):
        dst = features[i-1]
        dst_height, dst_width = feature_sizes[i-1]
        target_height = normalize_target_size(dst_height, scale_factor, output_size[0])

        if dst_height != dst_width:
            target_width = normalize_target_size(dst_width, scale_factor, output_size[1])
        else:
            target_width = target_height

        x = __transition_up_block(
            filters=num_classes,
            target_size=(target_height, target_width),
            upsampling_type='deconv',
            merge_type='add',
            block_prefix='feature%d' % (i+1)
        )([x, dst])

    if use_activation:
        output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        x = Activation(output_activation, name='outputs')(x)
    return x

