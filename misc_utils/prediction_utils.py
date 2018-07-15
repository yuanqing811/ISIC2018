import numpy as np


def inv_sigmoid(x):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1. - x))


eps = np.finfo(np.float32).eps


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def rot90_4D(images):
    return np.transpose(np.flip(images, axis=2), axes=[0, 2, 1, 3])


def rot180_4D(images):
    return np.flip(np.flip(images, axis=1), axis=2)


def rot270_4D(images):
    return np.flip(np.transpose(images, axes=[0, 2, 1, 3]), axis=2)


def fliplr_4D(images):
    return np.flip(images, axis=2)


def flipud_4D(images):
    return np.flip(images, axis=1)


def cyclic_pooling(y,
                   y_rot90,
                   y_rot180,
                   y_rot270,
                   y_fliplr=None,
                   y_rot90_fliplr=None,
                   y_rot180_fliplr=None,
                   y_rot270_fliplr=None,
                   use_sigmoid=True,
                   data_type='img'):

    if len(y.shape) == 3:
        data_type = 'mask'
        y = y[..., None]
        y_rot90 = y_rot90[..., None]
        y_rot180 = y_rot180[..., None]
        y_rot270 = y_rot270[..., None]
        y_fliplr = y_fliplr[..., None]
        y_rot90_fliplr = y_rot90_fliplr[..., None]
        y_rot180_fliplr = y_rot180_fliplr[..., None]
        y_rot270_fliplr = y_rot270_fliplr[..., None]

    y1 = rot270_4D(y_rot90)
    y2 = rot180_4D(y_rot180)
    y3 = rot90_4D(y_rot270)

    y4 = fliplr_4D(y_fliplr)
    y5 = rot270_4D(fliplr_4D(y_rot90_fliplr))
    y6 = rot180_4D(fliplr_4D(y_rot180_fliplr))
    y7 = rot90_4D(fliplr_4D(y_rot270_fliplr))

    y_stacked = np.stack([y, y1, y2, y3, y4, y5, y6, y7], axis=0)

    if use_sigmoid:
        y_stacked = inv_sigmoid(y_stacked)

    y_stacked = np.mean(y_stacked, axis=0, keepdims=False)
    y_stacked = sigmoid(y_stacked)

    if data_type == 'mask':
        y_stacked = y_stacked[..., 0]

    return y_stacked


def cyclic_stacking(x):
    assert len(x.shape) == 4
    x_rot90 = rot90_4D(x)
    x_rot180 = rot180_4D(x)
    x_rot270 = rot270_4D(x)

    x_fliplr = fliplr_4D(x)
    x_rot90_fliplr = fliplr_4D(x_rot90)
    x_rot180_fliplr = fliplr_4D(x_rot180)
    x_rot270_fliplr = fliplr_4D(x_rot270)

    return x, x_rot90, x_rot180, x_rot270, x_fliplr, x_rot90_fliplr, x_rot180_fliplr, x_rot270_fliplr