import keras
from keras.applications import inception_v3 as keras_inception_v3
from models import Backbone


class InceptionBackbone(Backbone):

    def __init__(self, backbone_name='inception_v3', **kwargs):
        super(InceptionBackbone, self).__init__(backbone_name)
        self.custom_objects['keras_inception_v3'] = keras_inception_v3

    def build_base_model(self, inputs, **kwargs):
        # create the inception backbone
        if self.backbone_name == 'inception_v3':
            inputs = keras.layers.Lambda(lambda x: keras_inception_v3.preprocess_input(x))(inputs)
            inception = keras_inception_v3.InceptionV3(include_top=False,
                                                       input_tensor=inputs,
                                                       weights='imagenet')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return inception

    def classification_model(self,
                             num_dense_layers=0,
                             num_dense_units=0,
                             dropout_rate=0.2,
                             pooling='avg',
                             name='default_inception_classification_model',
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """

        return super(InceptionBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                                   num_dense_units=num_dense_units,
                                                                   dropout_rate=dropout_rate,
                                                                   pooling=pooling,
                                                                   name=name, **kwargs)

    def segmentation_model(self,
                           input_padding=37,
                           init_nb_filters=64,
                           growth_rate=2,
                           nb_layers_per_block=2,
                           max_nb_filters=512,
                           upsampling_type='deconv',
                           name='default_inception_segmentation_model',
                           **kwargs):

        backbone_layer_names = kwargs.get('backbone_layer_names', None)

        if self.backbone_name == 'inception_v3':
            if backbone_layer_names is None:
                backbone_layer_names = ['activation_3',
                                        'activation_5',
                                        'mixed1',
                                        'mixed7',
                                        'mixed10']
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return super(InceptionBackbone, self).segmentation_model(input_padding=input_padding,
                                                                 init_nb_filters=init_nb_filters,
                                                                 growth_rate=growth_rate,
                                                                 nb_layers_per_block=nb_layers_per_block,
                                                                 max_nb_filters=max_nb_filters,
                                                                 upsampling_type=upsampling_type,
                                                                 backbone_layer_names=backbone_layer_names,
                                                                 name=name,
                                                                 **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inception_v3', ]

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))

