import os
from models import backbone
from paths import model_data_dir

import keras
from keras.models import load_model, model_from_json
from misc_utils.filename_utils import get_weights_filename
from misc_utils.filename_utils import get_json_filename

from metrics import dice_coeff
from metrics import jaccard_index
from metrics import class_jaccard_index
from metrics import pixelwise_precision
from metrics import pixelwise_sensitivity
from metrics import pixelwise_specificity
from metrics import pixelwise_recall

from losses import focal_loss, balanced_crossentropy, f1_loss

from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy

from keras.metrics import binary_accuracy
from keras.metrics import categorical_accuracy

from keras.optimizers import Adam


def load_model_weights_from(model, weights, skip_mismatch):
    if weights is None:
        return

    if os.path.exists(weights):
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        return

    weights_path = get_weights_filename(weights)
    if os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True, skip_mismatch=skip_mismatch)
        return

    raise ValueError('Unknown weights to load from!', weights, weights_path)


def save_model_to_run(model, run_name):
    json_path = get_json_filename(run_name)
    h5_path = get_weights_filename(run_name)

    with open(json_path, 'w') as json_file:
        json_file.write(model.to_json())

    model.save_weights(h5_path)


def load_model_from_run(backbone_name,
                        load_model_from,
                        load_weights_from=None,
                        skip_mismatch=True):
    b = backbone(backbone_name)
    json_path = get_json_filename(load_model_from)

    if not os.path.exists(json_path):
        h5_path = get_weights_filename(load_model_from)
        if not os.path.exists(h5_path):
            raise ValueError("run with name %s doesn't exist" % load_model_from)

        model = load_model(h5_path, custom_objects=b.custom_objects, compile=False)
    else:
        with open(json_path, 'r') as json_file:
            json_string = json_file.read()
        model = model_from_json(json_string, custom_objects=b.custom_objects)

        if load_weights_from:
            h5_path = get_weights_filename(load_weights_from)
            if os.path.exists(h5_path):
                model.load_weights(h5_path, by_name=True, skip_mismatch=skip_mismatch)
    return model


def freeze_model(model, layers=None):
    if layers is None:
        model.trainable = False
    else:
        for layer in layers:
            if isinstance(layer, int):
                layer = model.get_layer(index=layer)
            elif isinstance(layer, str):
                layer = model.get_layer(name=layer)
            else:
                raise ValueError('layer must be either an index or a string')
            layer.trainable = False

    return model


def plot_model(save_to_dir, model, name):
    filename = os.path.join(model_data_dir, save_to_dir, '%s.png' % name)
    keras.utils.plot_model(model,
                           to_file=filename,
                           show_shapes=True,
                           show_layer_names=True)


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def get_model_metrics(metrics, num_classes):

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
    return metrics


def get_model_loss(loss, num_classes, **kwargs):
    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = binary_crossentropy
            else:
                loss = categorical_crossentropy
        elif loss in {'fl', 'focal', 'focal_loss'}:
            alpha = kwargs.get('alpha', 0.75)
            gamma = kwargs.get('gamma', 2.0)
            loss = focal_loss(alpha=alpha, gamma=gamma, num_classes=num_classes)
        elif loss in {'bce', 'balanced_ce', 'balanced_crossentropy'}:
            alpha = kwargs.get('alpha', 0.75)
            loss = balanced_crossentropy(alpha=alpha, num_classes=num_classes)
        elif loss == 'dice':
            loss = dice_coeff(num_classes=num_classes)
        elif loss == 'f1':
            loss = f1_loss(num_classes=num_classes)
        else:
            raise ValueError('unknown loss %s' % loss)
    return loss


def compile_model(model, num_classes, metrics, loss, lr):
    metrics = get_model_metrics(metrics, num_classes=num_classes)
    loss = get_model_loss(loss, num_classes)
    model.compile(optimizer=Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)
