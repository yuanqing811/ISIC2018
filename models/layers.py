import keras
import backend as K
from keras.utils import conv_utils
from keras.engine import InputSpec


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    def __init__(self, scale_factor=(2, 2), target_size=None, data_format=None, method='bilinear', **kwargs):
        super(UpsampleLike, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_size = conv_utils.normalize_tuple(target_size, 2, 'target_size')
        self.scale_factor = conv_utils.normalize_tuple(scale_factor, 2, 'scale_factor')
        self.method = method
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, **kwargs):
        input_shape = keras.backend.get_variable_shape(inputs)
        if K.image_data_format() == 'channels_last':
            src_height, src_width = input_shape[1:3]
        else:
            src_height, src_width = input_shape[2:4]

        if self.target_size is not None:
            target_size = (src_height, src_width)
            while target_size[0] < self.target_size[0] or \
                    target_size[1] < self.target_size[1]:
                target_size = (target_size[0] * self.scale_factor[0],
                               target_size[1] * self.scale_factor[1])

        else:
            target_size = (src_height * self.scale_factor[0],
                           src_width * self.scale_factor[1])

        return K.resize_images(inputs, target_size, method=self.method)

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_last':
            src_height, src_width = input_shape[1:3]
        else:
            src_height, src_width = input_shape[2:4]

        target_size = (src_height * self.scale_factor[0], src_width * self.scale_factor[1])

        if self.target_size is not None:
            while target_size[0] < self.target_size[0] or \
                    target_size[1] < self.target_size[1]:
                target_size = (target_size[0] * self.scale_factor[0],
                               target_size[1] * self.scale_factor[1])

        return (input_shape[0],) + target_size + (input_shape[-1],)


class SubPixelUpscaling(keras.layers.Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = K.depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))