import os
from models import backbone
from paths import model_data_dir

import keras
from keras.models import load_model, model_from_json
from misc_utils.filename_utils import get_weights_filename
from misc_utils.filename_utils import get_json_filename


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


# def print_model_summary(save_to_dir, model, name):
#     filename = os.path.join(model_data_dir, save_to_dir, '%s.txt' % name)
#     with open(filename, 'w') as file:
#         model.summary(print_fn=lambda x: file.write(x + '\n'))

def plot_model(save_to_dir, model, name):
    filename = os.path.join(model_data_dir, save_to_dir, '%s.png' % name)
    keras.utils.plot_model(model,
                           to_file=filename,
                           show_shapes=True,
                           show_layer_names=True)


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None