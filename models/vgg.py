from models import Backbone

from keras.applications import vgg16 as keras_vgg16
from keras.applications import vgg19 as keras_vgg19

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

from keras.models import Model
from keras import backend as K
from keras.utils import conv_utils
from .ops import rot90_4D
from keras.engine.topology import get_source_inputs
from misc_utils.model_utils import name_or_none


class VGGBackbone(Backbone):
    def __init__(self, backbone_name='vgg16', **kwargs):
        super(VGGBackbone, self).__init__(backbone_name, **kwargs)

        if self.backbone_name == 'vgg16':
            self.custom_objects['keras_vgg16'] = keras_vgg16
        elif self.backbone_name == 'vgg19':
            self.custom_objects['keras_vgg19'] = keras_vgg19

    def build_base_model(self,
                         inputs,
                         nb_blocks=5,
                         nb_layers_per_block=2,
                         init_nb_filters=64,
                         growth_rate=2,
                         max_nb_filters=512,
                         activation='relu',
                         batch_normalization=False):

        # create the vgg backbone
        if self.backbone_name == 'vgg16':
            inputs = Lambda(lambda x: keras_vgg16.preprocess_input(x))(inputs)
            base_model = keras_vgg16.VGG16(input_tensor=inputs,
                                           include_top=False,
                                           weights='imagenet')
        elif self.backbone_name == 'vgg19':
            inputs = Lambda(lambda x: keras_vgg19.preprocess_input(x))(inputs)
            base_model = keras_vgg19.VGG19(input_tensor=inputs,
                                           include_top=False,
                                           weights='imagenet')
        elif self.backbone_name == 'unet':
            x = Lambda(lambda x: x / 255.)(inputs)
            base_model = encoder(input_tensor=x,
                                 init_nb_filters=init_nb_filters,
                                 growth_rate=growth_rate,
                                 nb_blocks=nb_blocks,
                                 nb_layers_per_block=nb_layers_per_block,
                                 max_nb_filters=max_nb_filters,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 name='unet')
        else:
            raise NotImplementedError("Backbone '{}' not recognized.".format(self.backbone_name))

        return base_model

    def classification_model(self,
                             num_dense_layers=2,
                             num_dense_units=1024,
                             dropout_rate=0.,
                             pooling=None,
                             name='default_vgg_classification_model',
                             **kwargs):

        return super(VGGBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                             num_dense_units=num_dense_units,
                                                             dropout_rate=dropout_rate,
                                                             pooling=pooling,
                                                             name=name,
                                                             **kwargs)

    def segmentation_model(self,
                           nb_blocks=5,
                           init_nb_filters=64,
                           growth_rate=2,
                           nb_layers_per_block=2,
                           max_nb_filters=512,
                           upsampling_type='deconv',
                           activation='relu',
                           name='default_vgg_segmentation_model',
                           **kwargs):

        if self.backbone_name == 'vgg16':
            backbone_layer_names = ['block1_conv2',
                                    'block2_conv2',
                                    'block3_conv3',
                                    'block4_conv3',
                                    'block5_conv3']
        elif self.backbone_name == 'vgg19':
            backbone_layer_names = ['block1_conv2',
                                    'block2_conv2',
                                    'block3_conv4',
                                    'block4_conv4',
                                    'block5_conv4']
        elif self.backbone_name == 'unet':
            nb_blocks = self.backbone_options.get('nb_blocks', 5)
            nb_layers_per_block = conv_utils.normalize_tuple(nb_layers_per_block,
                                                             nb_blocks,
                                                             'nb_layers_per_block')
            backbone_layer_names = ['block%d_activation%d' % (i + 1, nb_layers_per_block[i])
                                    for i in range(nb_blocks)]
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return super(VGGBackbone, self).segmentation_model(init_nb_filters=init_nb_filters,
                                                           growth_rate=growth_rate,
                                                           nb_layers_per_block=nb_layers_per_block,
                                                           max_nb_filters=max_nb_filters,
                                                           backbone_layer_names=backbone_layer_names,
                                                           upsampling_type=upsampling_type,
                                                           activation=activation,
                                                           name=name,
                                                           **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'unet']
        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))


def __conv_block(nb_filters,
                 dilation_rate=1,
                 nb_layers=2,
                 activation='relu',
                 cyclic_rolling=False,
                 batch_normalization=False,
                 block_prefix='conv_block',
                 **options):

    if isinstance(nb_filters, int):
        nb_filters = conv_utils.normalize_tuple(nb_filters, nb_layers, 'nb_filters')
    elif isinstance(nb_filters, tuple) or isinstance(nb_filters, list):
        nb_layers = len(nb_filters)
    else:
        raise ValueError('nb_filters must be either an int or tuple / list of ints')

    dilation_rate = conv_utils.normalize_tuple(dilation_rate, nb_layers, 'dilation_rate')

    convs = [Conv2D(filters=nb_filters[i],
                    dilation_rate=dilation_rate[i],
                    activation='linear',
                    name=name_or_none(block_prefix, '_conv%d' % (i+1)),
                    **options)
             for i in range(nb_layers)]

    if batch_normalization:
        bns = [BatchNormalization(name=name_or_none(block_prefix, '_bn%d' % i)) for i in range(nb_layers)]

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
            for i, n in enumerate(nb_filters):
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
            init_nb_filters=64,
            growth_rate=2,
            nb_blocks=5,
            nb_layers_per_block=2,
            max_nb_filters=512,
            activation='relu',
            batch_normalization=False,
            name='default_encoder_model'):

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        # 'kernel_initializer': Orthogonal(gain=1.0, seed=None),
        # 'bias_initializer': Zeros(),
    }

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input
    nb_layers_per_block = conv_utils.normalize_tuple(nb_layers_per_block,
                                                     nb_blocks,
                                                     'nb_layers_per_block')
    for i in range(nb_blocks):
        nb_filters = int(init_nb_filters * (growth_rate ** i))
        nb_filters = min(nb_filters, max_nb_filters)
        block_prefix = 'block%d' % (i+1)
        nb_filters = conv_utils.normalize_tuple(nb_filters,
                                                nb_layers_per_block[i],
                                                'nb_filters')
        x = __conv_block(nb_filters=nb_filters,
                         activation=activation,
                         block_prefix=block_prefix,
                         batch_normalization=batch_normalization,
                         **options)(x)

        if i < nb_blocks-1:
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                             name='%s_pool' % block_prefix)(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs=inputs, outputs=x, name=name)
    return model
