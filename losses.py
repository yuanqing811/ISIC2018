from keras import backend as K
import tensorflow as tf

'''
The following code is adated from 
https://github.com/fizyr/keras-retinanet
'''


def balanced_crossentropy(alpha=0.5, num_classes=1):
    def balanced_binary_crossentropy(y_true, y_pred):
        fg = K.greater_equal(y_true, 0.5)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(fg, alpha_factor, 1. - alpha_factor)
        loss = alpha_factor * K.binary_crossentropy(y_true, y_pred)
        normalizer = tf.count_nonzero(fg, dtype=K.floatx())
        return K.sum(loss)/K.maximum(normalizer, 1.)
        # return K.mean(loss)

    def balanced_categorical_crossentropy(y_true, y_pred):
        fg = K.greater_equal(y_true, 0.5)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(fg, alpha_factor, 1. - alpha_factor)
        loss = alpha_factor * K.categorical_crossentropy(y_true, y_pred)
        normalizer = tf.count_nonzero(fg, dtype=K.floatx())
        return K.sum(loss)/normalizer

    if num_classes == 1:
        return balanced_binary_crossentropy
    else:
        return balanced_categorical_crossentropy


def focal_loss(alpha=0.25, gamma=2.0, num_classes=1):
    """
    α-balanced CE loss
    α ∈ [0, 1] for class 1 and 1 - α for class 0.
    CE(p_t) = -α_t log(p_t)

    focal loss
    FL(p_t) = −(1 − p_t)γ log(p_t). γ ∈ [0, 5]

    α-balanced variant of the focal loss
    FL(pt) = −α_t (1 − p_t)**γ log(pt).
    """
    def binary_focal_loss(y_true, y_pred):
        # compute the focal loss
        # CE(p_t) = -log(p_t)
        # FL(p_t) = -(1 - p_t) ** gamma * log(p_t)
        # if y = 1, CE(p, y) = - log(p); otherwise, CE(p, y) = -log(1-p)

        fg = K.greater_equal(y_true, 0.5)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(fg, alpha_factor, 1. - alpha_factor)
        focal_weight = tf.where(fg, 1. - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
        return K.mean(loss)

        # compute the normalizer: the number of positive anchors
        # normalizer = tf.count_nonzero(fg, dtype=K.floatx())
        # return K.sum(loss) / K.maximum(normalizer, 1.)

    def categorical_focal_loss(y_true, y_pred):
        alpha_factor = K.ones_like(y_true) * alpha
        fg = K.greater_equal(y_true, 0.5)
        alpha_factor = tf.where(fg, alpha_factor, 1. - alpha_factor)
        focal_weight = tf.where(fg, 1. - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
        # normalizer = tf.count_nonzero(fg, axis=[1, 2], dtype=tf.float32)
        # loss = K.sum(loss, axis=[1, 2])/K.clip(normalizer, 1, None)
        return K.mean(loss)

    if num_classes == 1:
        return binary_focal_loss
    else:
        return categorical_focal_loss


def smooth_f1(num_classes=1):
    raise NotImplementedError
