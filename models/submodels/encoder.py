from keras import Input, backend as K, Model
from keras.engine import get_source_inputs
from keras.layers import Conv2D, LeakyReLU, Lambda, Concatenate, MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.utils import conv_utils

from misc_utils.model_utils import name_or_none
from models.ops import rot90_4D


def __conv_block(filters,
                 activation='relu',
                 kernel_size=3,
                 strides=1,
                 dilation_rate=1,
                 cyclic_rolling=False,
                 batch_normalization=False,
                 block_prefix='conv_block',
                 **options):

    nb_layers = len(filters)

    convs = [Conv2D(filters[i],
                    conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size'),
                    strides=strides,
                    dilation_rate=conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate'),
                    activation=None,
                    padding='same',
                    name=name_or_none(block_prefix, '_conv{a}-k{b}-d{c}'.format(a=(i+1),
                                                                                b=kernel_size,
                                                                                c=dilation_rate)))
             for i in range(nb_layers)]

    if batch_normalization:
        bns = [BatchNormalization(name=name_or_none(block_prefix, '_bn%d' % i))
               for i in range(nb_layers)]

    if activation in {'relu', }:
        activations = [Activation(activation=activation,
                                  name='%s_activation%d' % (block_prefix, i+1))
                       for i in range(nb_layers)]

    elif activation == 'leakyReLU':
        alpha = options.get('alpha', 0.33)
        activations = [LeakyReLU(alpha=alpha,
                                 name='%s_activation%d' % (block_prefix, i + 1))
                       for i in range(nb_layers)]
    else:
        raise NotImplementedError

    def block(x):
        if cyclic_rolling:
            rot0 = Lambda(lambda _x: rot90_4D(_x, 0), name='%s_rot0_1' % block_prefix)
            rot90 = Lambda(lambda _x: rot90_4D(_x, 1), name='%s_rot90_1' % block_prefix)
            rot180 = Lambda(lambda _x: rot90_4D(_x, 2), name='%s_rot180_1' % block_prefix)
            rot270 = Lambda(lambda _x: rot90_4D(_x, 3), name='%s_rot270_1' % block_prefix)
            xs = [rot0(x), rot90(x), rot180(x), rot270(x)]
        else:
            xs = [x, ]

        x_rot = []
        for _x in xs:
            for i, n in enumerate(filters):
                _x = convs[i](_x)
                if batch_normalization:
                    _x = bns[i](_x)
                _x = activations[i](_x)
            x_rot.append(_x)

        if cyclic_rolling:
            # cyclic stacking in the feature space
            rot0 = Lambda(lambda _x: rot90_4D(_x, 0), name='%s_rot0_2' % block_prefix)
            rot90 = Lambda(lambda _x: rot90_4D(_x, 1), name='%s_rot90_2' % block_prefix)
            rot180 = Lambda(lambda _x: rot90_4D(_x, 2), name='%s_rot180_2' % block_prefix)
            rot270 = Lambda(lambda _x: rot90_4D(_x, 3), name='%s_rot270_2' % block_prefix)
            x = Concatenate(axis=-1, name='%s_conc' % block_prefix)([rot0(x_rot[0]),
                                                                     rot270(x_rot[1]),
                                                                     rot180(x_rot[2]),
                                                                     rot90(x_rot[3])])
        else:
            x = x_rot[0]
        return x
    return block


def encoder(input_tensor=None,
            input_shape=(224, 224, 3),
            layers_per_block=2,
            blocks=(64, 128, 256, 512, 1024),
            activation='relu',
            dilation_rate=1,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            batch_normalization=False,
            name='default_encoder_model'):

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input
    nb_blocks = len(blocks)
    layers_per_block = conv_utils.normalize_tuple(layers_per_block, nb_blocks, 'layers_per_block')
    dilation_rate = conv_utils.normalize_tuple(dilation_rate, nb_blocks, 'dilation_rate')

    for i in range(nb_blocks):
        block_prefix = 'block%d' % (i+1)
        nb_filters = conv_utils.normalize_tuple(blocks[i], layers_per_block[i], 'nb_filters')

        x = __conv_block(filters=nb_filters,
                         activation=activation,
                         dilation_rate=dilation_rate[i],
                         batch_normalization=batch_normalization,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         block_prefix=block_prefix)(x)

        if i < nb_blocks-1:
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='%s_pool' % block_prefix)(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs=inputs, outputs=x, name=name)
    return model
