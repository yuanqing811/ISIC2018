from models import Backbone

from keras.applications import vgg16 as keras_vgg16
from keras.applications import vgg19 as keras_vgg19

from keras.layers import Lambda

from keras.utils import conv_utils
from models.submodels.encoder import encoder


class VGGBackbone(Backbone):
    def __init__(self, backbone_name='vgg16', **kwargs):
        super(VGGBackbone, self).__init__(backbone_name, **kwargs)

        if self.backbone_name == 'vgg16':
            self.custom_objects['keras_vgg16'] = keras_vgg16
        elif self.backbone_name == 'vgg19':
            self.custom_objects['keras_vgg19'] = keras_vgg19

    def build_base_model(self,
                         inputs,
                         blocks=(64, 128, 256, 512, 1024),
                         layers_per_block=2,
                         activation='relu',
                         dilation_rate=1,
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
            inputs = Lambda(lambda x: x / 127.5 - 1.)(inputs)
            base_model = encoder(input_tensor=inputs,
                                 blocks=blocks,
                                 layers_per_block=layers_per_block,
                                 activation=activation,
                                 dilation_rate=dilation_rate,
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
                           blocks=(64, 128, 256, 512, 512),
                           layers_per_block=2,
                           upsampling_type='deconv',
                           activation='relu',
                           dilation_rate=1,
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
            nb_blocks = len(blocks)
            layers_per_block = conv_utils.normalize_tuple(layers_per_block, nb_blocks, 'layers_per_block')
            backbone_layer_names = ['block%d_activation%d' % (i + 1, layers_per_block[i])
                                    for i in range(nb_blocks)]
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return super(VGGBackbone, self).segmentation_model(blocks=blocks,
                                                           layers_per_block=layers_per_block,
                                                           backbone_layer_names=backbone_layer_names,
                                                           upsampling_type=upsampling_type,
                                                           activation=activation,
                                                           dilation_rate=dilation_rate,
                                                           name=name,
                                                           **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'unet']
        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))


