import keras
from keras import backend as K
import numpy as np
import tensorflow as tf


def resize_images(x, height_factor, width_factor, data_format):
    """Resizes the images contained in a 4D tensor.

    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = K.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Unknown data_format: ' + str(data_format))


class UpsampleLike(keras.layers.Layer):
    """
    Adapted from https://github.com/fizyr/keras-retinanet
    """
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)

        if K.image_data_format() == 'channels_last':
            source_height, source_width = source_shape[1:3]
            target_height, target_width = target_shape[1:3]
        else:
            source_height, source_width = source_shape[2:4]
            target_height, target_width = target_shape[2:4]

        return K.resize_images(source, target_height / source_height,
                               target_width / source_width,
                               K.image_data_format())

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
