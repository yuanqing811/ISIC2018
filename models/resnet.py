import keras
from models import Backbone
from keras.applications import resnet50 as keras_resnet50


class ResNetBackbone(Backbone):
    def __init__(self, backbone_name='resnet50'):
        super(ResNetBackbone, self).__init__(backbone_name)
        self.custom_objects['keras_resnet50'] = keras_resnet50

    def build_base_model(self, inputs, **kwarg):
        # create the vgg backbone
        if self.backbone_name == 'resnet50':
            inputs = keras.layers.Lambda(lambda x: keras_resnet50.preprocess_input(x))(inputs)
            resnet = keras_resnet50.ResNet50(input_tensor=inputs,
                                             include_top=False,
                                             weights='imagenet')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone_name))

        return resnet

    def classification_model(self,
                             num_dense_layers=0,
                             num_dense_units=0,
                             dropout_rate=0.2,
                             pooling='avg',
                             name='default_resnet_classification_model',
                             **kwargs):
        """ Returns a classifier model using the correct backbone.
        """
        return super(ResNetBackbone, self).classification_model(num_dense_layers=num_dense_layers,
                                                                num_dense_units=num_dense_units,
                                                                dropout_rate=dropout_rate,
                                                                pooling=pooling,
                                                                name=name,
                                                                **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', ]

        if self.backbone_name not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name,
                                                                                       allowed_backbones))
