from models import Backbone
import keras
from keras.applications import densenet as keras_densenet


class DenseNetBackbone(Backbone):
    def __init__(self, backbone_name='densenet121', **kwargs):
        super(DenseNetBackbone, self).__init__(backbone_name, **kwargs)
        self.custom_objects['keras_densenet'] = keras_densenet

    def build_base_model(self, inputs, blocks=None):
        inputs = keras.layers.Lambda(lambda x: keras_densenet.preprocess_input(x))(inputs)

        if self.backbone_name == 'densenet121':
            densenet = keras_densenet.DenseNet121(include_top=False,
                                                  weights='imagenet',
                                                  input_tensor=inputs)
        elif self.backbone_name == 'densenet169':
            densenet = keras_densenet.DenseNet169(include_top=False,
                                                  input_tensor=inputs,
                                                  weights='imagenet')
        elif self.backbone_name == 'densenet201':
            densenet = keras_densenet.DenseNet201(include_top=False,
                                                  input_tensor=inputs,
                                                  weights='imagenet')
        elif self.backbone_name == 'densenet':
            if blocks is None:
                raise ValueError('blocks must be specified to use custom densenet backbone')

            densenet = keras_densenet.DenseNet(blocks=blocks,
                                               include_top=False,
                                               input_tensor=inputs,
                                               weights='imagenet')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return densenet

    def classification_model(self,
                             num_dense_layers=0,
                             num_dense_units=0,
                             dropout_rate=0.2,
                             pooling='avg',
                             name='default_inception_classification_model',
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """

        return super(DenseNetBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                                  num_dense_units=num_dense_units,
                                                                  dropout_rate=dropout_rate,
                                                                  pooling=pooling,
                                                                  name=name,
                                                                  **kwargs)

    def segmentation_model(self,
                           init_nb_filters=64,
                           growth_rate=2,
                           nb_layers_per_block=2,
                           max_nb_filters=512,
                           upsampling_type='deconv',
                           name='default_densenet_segmentation_model',
                           **kwargs):

        if self.backbone_name == 'densenet121':
            # blocks = [6, 12, 24, 16]
            backbone_layer_names = ['conv2_block6_concat',
                                    'conv3_block12_concat',
                                    'conv4_block24_concat',
                                    'conv5_block16_concat']
        elif self.backbone_name == 'densenet169':
            # blocks = [6, 12, 32, 32]
            backbone_layer_names = ['conv2_block6_concat',
                                    'conv3_block12_concat',
                                    'conv4_block32_concat',
                                    'conv5_block32_concat']
        elif self.backbone_name == 'densenet201':
            # blocks = [6, 12, 48, 32]
            backbone_layer_names = ['conv2_block6_concat',
                                    'conv3_block12_concat',
                                    'conv4_block48_concat',
                                    'conv5_block32_concat']
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return super(DenseNetBackbone, self).segmentation_model(init_nb_filters=init_nb_filters,
                                                                growth_rate=growth_rate,
                                                                nb_layers_per_block=nb_layers_per_block,
                                                                max_nb_filters=max_nb_filters,
                                                                backbone_layer_names=backbone_layer_names,
                                                                upsampling_type=upsampling_type,
                                                                name=name,
                                                                **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['densenet121',  'densenet169', 'densenet201']

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))


