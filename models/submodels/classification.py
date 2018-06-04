import keras
from keras import Input, backend as K, regularizers


def default_classification_model(input_shape=None,
                                 input_tensor=None,
                                 num_classes=7,
                                 num_dense_layers=2,
                                 num_dense_units=256,
                                 dropout_rate=0.,
                                 pooling=None,
                                 use_output_activation=True,
                                 kernel_regularizer=None):

    """
    :param kernel_regularizer: l1 or l2 or none regularization
    :param num_classes:             # of classes to predict a score for each feature level.
    :param input_shape:             Input shape
    :param input_tensor:            Input tensor
    :param num_dense_layers:         Number of dense layers before the output layer
    :param num_dense_units:              The number of filters to use in the layers in the classification submodel.
    :param dropout_rate:            Dropout Rate
    :param pooling:                 which pooling to use at conv output
    :param use_output_activation:   whether to use output activation
    :return: A keras.model.Model that predicts class
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    assert kernel_regularizer in [None, 'L1', 'L2', 'L1-L2'], \
        'Unknown regularizer %s' % kernel_regularizer

    if kernel_regularizer == 'L1':
        kernel_regularizer = regularizers.l1(1e-4)
    elif kernel_regularizer == 'L2':
        kernel_regularizer = regularizers.l2(1e-3)
    elif kernel_regularizer == 'L1-L2':
        kernel_regularizer = regularizers.l1_l2(l1=1e-4, l2=1e-3)

    assert pooling in {None, 'avg', 'max', 'flatten'}, 'Unknown pooling option %s' % pooling

    if pooling == 'avg':
        outputs = keras.layers.GlobalAveragePooling2D(name='avg_pool_our')(img_input)
    elif pooling == 'max':
        outputs = keras.layers.GlobalMaxPooling2D(name='max_pool_our')(img_input)
    else:
        outputs = keras.layers.Flatten(name='flatten_our')(img_input)

    if dropout_rate > 0.:
        outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)

    for i in range(num_dense_layers):
        outputs = keras.layers.Dense(num_dense_units,
                                     activation='relu',
                                     name='fc%d' % (i + 1),
                                     kernel_regularizer=kernel_regularizer
                                     )(outputs)

        outputs = keras.layers.Dropout(rate=dropout_rate)(outputs)

    outputs = keras.layers.Dense(num_classes,
                                 name='predictions',
                                 kernel_regularizer=kernel_regularizer
                                 )(outputs)

    if use_output_activation:
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        outputs = keras.layers.Activation(activation, name='outputs')(outputs)

    return outputs