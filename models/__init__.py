class Backbone(object):
    """
    This class stores additional information on backbones
    """

    def __init__(self, backbone_name, **kwargs):
        """
        :param backbone_name: name of the backbone
        :param kwargs: user provided kwargs in case a custom base model is needed
        """
        from metrics import dice_coeff
        from metrics import jaccard_index
        from metrics import pixelwise_precision
        from metrics import pixelwise_specificity
        from metrics import pixelwise_sensitivity
        from initializers import PriorProbability

        self.custom_objects = {
            # included custom metrics in case the saved model need to be compiled
            'dice_coeff': dice_coeff,
            'jaccard_index': jaccard_index,
            'pixelwise_precision': pixelwise_precision,
            'pixelwise_specificity': pixelwise_specificity,
            'pixelwise_sensitivity': pixelwise_sensitivity,
            'PriorProbability': PriorProbability,
        }

        self.backbone_name = backbone_name
        self.backbone_options = kwargs
        self.scale_factor = 2
        self.validate()

    def build_base_model(self, inputs, **kwarg):
        raise NotImplementedError('backbone method not implemented')

    def classification_model(self,
                             input_shape=None,
                             input_padding=None,
                             submodel=None,
                             num_classes=7,
                             num_dense_layers=2,
                             num_dense_units=256,
                             dropout_rate=0.,
                             pooling=None,
                             use_output_activation=True,
                             kernel_regularizer=None,
                             use_activation=True,
                             include_top=True,
                             name='default_classification_model',
                             print_model_summary=False,
                             plot_model_summary=False,
                             load_from=None,
                             load_model_from=None,
                             load_weights_from=None,
                             save_to=None,
                             lr=1e-5,
                             ):
        """ Returns a classifier model using the correct backbone.
        """
        import keras
        from keras import backend as K
        from models.submodels.classification import default_classification_model
        from misc_utils.model_utils import plot_model
        from misc_utils.model_utils import load_model_from_run
        from misc_utils.model_utils import save_model_to_run
        from misc_utils.model_utils import load_model_weights_from
        from misc_utils.print_utils import on_aws

        if load_from:
            model = load_model_from_run(self.backbone_name, load_from, load_from)
        elif load_model_from:
            model = load_model_from_run(self.backbone_name, load_model_from, load_weights_from)
        else:
            if K.image_data_format() == 'channels_last':
                input_shape = (224, 224, 3) if input_shape is None else input_shape
            else:
                input_shape = (3, 224, 224) if input_shape is None else input_shape

            inputs = keras.layers.Input(shape=input_shape)

            x = inputs
            if input_padding is not None:
                x = keras.layers.ZeroPadding2D(padding=input_padding)(x)

            base_model = self.build_base_model(inputs=x, **self.backbone_options)
            x = base_model.output

            if submodel is None:
                outputs = default_classification_model(input_tensor=x,
                                                       input_shape=base_model.output_shape[1:],
                                                       num_classes=num_classes,
                                                       num_dense_layers=num_dense_layers,
                                                       num_dense_units=num_dense_units,
                                                       dropout_rate=dropout_rate,
                                                       pooling=pooling,
                                                       use_output_activation=use_activation,
                                                       kernel_regularizer=kernel_regularizer)
            else:
                outputs = submodel(x)

            model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)

            if load_weights_from:
                load_model_weights_from(model, load_weights_from, skip_mismatch=True)

        if print_model_summary:
            model.summary()

        if plot_model_summary and not on_aws():
            plot_model(save_to_dir=save_to, model=model, name=name)

        if save_to:
            save_model_to_run(model, save_to)

        compile_model(model=model,
                      num_classes=num_classes,
                      metrics='acc',
                      loss='ce',
                      lr=lr)

        return model

    def segmentation_model(self,
                           load_from=None,
                           load_model_from=None,
                           load_weights_from=None,
                           save_to=None,
                           lr=1e-5,
                           loss='ce',
                           metrics=None,
                           print_model_summary=False,
                           plot_model_summary=False,
                           input_shape=None,
                           input_padding=None,
                           backbone_layer_names=None,
                           submodel=None,
                           modifier=None,
                           num_classes=1,
                           init_nb_filters=64,
                           growth_rate=2,
                           nb_layers_per_block=2,
                           max_nb_filters=512,
                           bottleneck=False,
                           upsampling_type='deconv',
                           activation='relu',
                           use_activation=True,
                           include_top=True,
                           prior_probability=0.5,
                           name='default_segmentation_model'):
        """
        Returns a segmentation model using the correct backbone
        """
        import keras
        from models.submodels.segmentation import default_decoder_model
        from misc_utils.model_utils import plot_model
        from misc_utils.model_utils import load_model_from_run
        from misc_utils.model_utils import save_model_to_run
        from misc_utils.model_utils import load_model_weights_from
        from misc_utils.print_utils import on_aws

        if load_from:
            print('loading from', load_from)
            model = load_model_from_run(self.backbone_name, load_from, load_from)
        elif load_model_from:
            model = load_model_from_run(self.backbone_name, load_model_from, load_weights_from)
        else:
            inputs = keras.layers.Input(shape=input_shape)
            if keras.backend.image_data_format() == 'channels_last':
                indices = slice(0, 2)
                input_shape = (224, 224, 3) if input_shape is None else input_shape
            else:
                indices = slice(1, 3)
                input_shape = (3, 224, 224) if input_shape is None else input_shape

            output_size = input_shape[indices]

            x = inputs
            if input_padding is not None:
                x = keras.layers.ZeroPadding2D(padding=input_padding)(x)

            base_model = self.build_base_model(inputs=x, **self.backbone_options)

            if backbone_layer_names:
                backbone_layers = [base_model.get_layer(name=layer_name) for layer_name in backbone_layer_names]
                backbone_features = [backbone_layer.output for backbone_layer in backbone_layers]
            else:
                outputs = base_model.output
                backbone_features = [outputs, ]

            if submodel is None:
                outputs = default_decoder_model(features=backbone_features,
                                                num_classes=num_classes,
                                                output_size=output_size,
                                                scale_factor=self.scale_factor,
                                                init_nb_filters=init_nb_filters,
                                                growth_rate=growth_rate,
                                                nb_layers_per_block=nb_layers_per_block,
                                                max_nb_filters=max_nb_filters,
                                                upsampling_type=upsampling_type,
                                                bottleneck=bottleneck,
                                                activation=activation,
                                                use_activation=use_activation,
                                                include_top=False)
            else:
                outputs = submodel(backbone_features)

            if include_top:
                outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='linear', name='predictions')(outputs)
                if use_activation:
                    output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
                    outputs = keras.layers.Activation(output_activation, name='outputs')(outputs)

            model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
            if load_weights_from:
                load_model_weights_from(model, load_weights_from, skip_mismatch=True)

        if print_model_summary:
            model.summary()

        if plot_model_summary and save_to and not on_aws():
            plot_model(save_to_dir=save_to, model=model, name=name)

        if save_to:
            save_model_to_run(model, save_to)

        if modifier:
            model = modifier(model)

        if metrics is None:
            metrics = ['acc',
                       'jaccard_index',
                       'pixelwise_sensitivity',
                       'pixelwise_specificity']

        compile_model(model=model,
                      num_classes=num_classes,
                      metrics=metrics,
                      loss=loss,
                      lr=lr)

        return model

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('backbone method not implemented')


def backbone(backbone_name, **kwargs):
    """
    Returns a backbone object for the given backbone.
    """
    if 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'unet' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'inception' in backbone_name:
        from .inception import InceptionBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return b(backbone_name, **kwargs)


def compile_model(model, num_classes, metrics, loss, lr):
    from keras.losses import binary_crossentropy
    from keras.losses import categorical_crossentropy

    from keras.metrics import binary_accuracy
    from keras.metrics import categorical_accuracy

    from keras.optimizers import Adam

    from metrics import dice_coeff
    from metrics import jaccard_index
    from metrics import class_jaccard_index
    from metrics import pixelwise_precision
    from metrics import pixelwise_sensitivity
    from metrics import pixelwise_specificity
    from metrics import pixelwise_recall

    from losses import focal_loss

    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = binary_crossentropy
            else:
                loss = categorical_crossentropy
        elif loss in {'focal', 'focal_loss'}:
            loss = focal_loss(num_classes)
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = binary_accuracy if num_classes == 1 else categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_coeff':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)

