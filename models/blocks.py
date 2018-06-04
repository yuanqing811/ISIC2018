import tensorflow as tf
from keras.layers import Concatenate
from keras.layers import Lambda
from models.ops import rot90_4D


def cyclic_slicing(block_prefix='cyclic_slicing'):
    def block(x):
        rot0 = Lambda(lambda _x: rot90_4D(_x, 0), name='%s_rot0' % block_prefix)
        rot90 = Lambda(lambda _x: rot90_4D(_x, 1), name='%s_rot90' % block_prefix)
        rot180 = Lambda(lambda _x: rot90_4D(_x, 2), name='%s_rot180' % block_prefix)
        rot270 = Lambda(lambda _x: rot90_4D(x, 3), name='%s_rot270' % block_prefix)
        x = Concatenate(axis=0, name='%s_conc' % block_prefix)([rot0(x), rot90(x), rot180(x), rot270(x)])
        return x
    return block


def cyclic_pooling_4D(block_prefix='cyclic_pooling'):
    def block(x):
        x = Lambda(lambda _x: tf.split(_x, 4, axis=0),
                   name='%s_split' % block_prefix)(x)

        x = Lambda(lambda _x: tf.sqrt(rot90_4D(_x[0], 0) ** 2 +
                                      rot90_4D(_x[1], 3) ** 2 +
                                      rot90_4D(_x[2], 2) ** 2 +
                                      rot90_4D(_x[3], 1) ** 2)/4,
                   name='%s_rms' % block_prefix)(x)
        return x
    return block


def cyclic_pooling_2D(block_prefix='cyclic_pooling'):
    def block(x):
        x = Lambda(lambda _x: tf.split(_x, 4, axis=0),
                   name='%s_split' % block_prefix)(x)
        x = Lambda(lambda _x: tf.sqrt(_x[0] ** 2 +
                                      _x[1] ** 2 +
                                      _x[2] ** 2 +
                                      _x[3] ** 2)/4,
                   name='%s_rms' % block_prefix)(x)
        return x
    return block


