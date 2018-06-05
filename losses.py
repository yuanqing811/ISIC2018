from keras import backend as K
import tensorflow as tf

'''
The following code is adated from 
https://github.com/fizyr/keras-retinanet
'''
def focal_loss(alpha=0.25, gamma=2.0, num_classes=1):
    def binary_focal_loss(y_true, y_pred):
        # # # filter out "ignore" anchors
        # anchor_state = K.max(y_true, axis=2)  # -1 for ignore, 0 for background, 1+ for objects
        # indices = tf.where(K.not_equal(anchor_state, -1))
        # y_true = tf.gather_nd(y_true, indices)
        # y_pred = tf.gather_nd(y_pred, indices)

        # compute the focal loss
        # CE(p_t) = -log(p_t)
        # FL(p_t) = -(1 - p_t) ** gamma * log(p_t)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
        return loss

        # # compute the normalizer: the number of positive anchors
        # normalizer = tf.where(K.equal(anchor_state, 1))
        # normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        # normalizer = K.maximum(1., normalizer)
        # return K.sum(cls_loss) / normalizer

    def categorical_focal_loss(y_true, y_pred):
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = tf.where(K.equal(y_true, 1), 1. - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        loss = focal_weight * K.categorical_crossentropy(y_true, y_pred)
        normalizer = K.sum(K.abs(y_true), axis=[1, 2])
        loss = K.sum(loss, axis=[1, 2])/K.maximum(1., normalizer)
        return K.mean(loss)

    if num_classes == 1:
        return binary_focal_loss
    else:
        return categorical_focal_loss
