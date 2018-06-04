from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from keras import backend as K


def _rot90(images):
    return K.permute_dimensions(K.reverse(images, [2]), [0, 2, 1, 3])


def _rot180(images):
    return K.reverse(images, [1, 2])


def _rot270(images):
    return K.reverse(K.permute_dimensions(images, [0, 2, 1, 3]), [2])


def rot90_4D(images, k):
    """Rotate batch of images counter-clockwise by 90 degrees `k` times.
    Args:
      images: 4-D Tensor of shape `[height, width, channels]`.
      k: A scalar integer. The number of times the images are rotated by 90
        degrees.
      name_scope: A valid TensorFlow name scope.
    Returns:
      A 4-D tensor of the same type and shape as `images`.
    """

    def _rot90():
        return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])

    def _rot180():
        return array_ops.reverse_v2(images, [1, 2])

    def _rot270():
        return array_ops.reverse_v2(array_ops.transpose(images, [0, 2, 1, 3]), [2])

    cases = [(math_ops.equal(k, 1), _rot90),
             (math_ops.equal(k, 2), _rot180),
             (math_ops.equal(k, 3), _rot270)]

    result = control_flow_ops.case(
        cases, default=lambda: images, exclusive=True)

    shape = result.get_shape()
    result.set_shape([shape[0], None, None, shape[3]])
    return result