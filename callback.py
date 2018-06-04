import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.utils.vis_utils import plot_model
from misc_utils.eval_utils import get_confusion_matrix, get_precision_recall
from misc_utils.filename_utils import get_weights_filename, get_csv_filename

plt.ion()


class PlotModel(Callback):
    def __init__(self, filename):
        super(PlotModel, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs=None):
        plot_model(self.model,
                   to_file=self.filename,
                   show_shapes=True,
                   show_layer_names=True)


class ModelSummary(Callback):
    def __init__(self, filename):
        self.filename = filename
        super(ModelSummary, self).__init__()

    def on_train_begin(self, logs=None):
        with open(self.filename, 'w') as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))


class ValidationPrediction(Callback):
    def __init__(self, show_confusion_matrix=False, **kwargs):
        super(ValidationPrediction, self).__init__()

        self.show_confusion_matrix = show_confusion_matrix

        self.visualize = kwargs.get('visualize', False)
        self.nrows = kwargs.get('nrows', 5)
        self.ncols = kwargs.get('ncols', 5)
        self.mask_colors = kwargs.get('mask_colors', ['r', 'b', 'g', 'c', 'm', 'y'])
        self.n_choices = self.nrows * self.ncols

        # for display purposes
        self.fig = None
        self.ax = None
        self.indices = None

        self.confusion_fig = None
        self.confusion_ax = None

        # setup
        self.y_true = None
        self.y_pred = None

    def on_epoch_end(self, epoch, logs=None):
        self.make_predictions()
        if self.show_confusion_matrix:
            self.view_confusion_matrix()

        if self.visualize:
            self.visualize_validation_prediction()

    def make_predictions(self):
        self.y_pred = self.model.predict(self.validation_data[0])
        self.y_true = self.validation_data[1]

    def view_confusion_matrix(self):
        _ = get_confusion_matrix(y_true=self.y_true, y_pred=self.y_pred, print_cm=True)
        get_precision_recall(y_true=self.y_true, y_pred=self.y_pred)

    def visualize_validation_prediction(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5),
                                             nrows=self.nrows,
                                             ncols=self.ncols,
                                             sharex='all',
                                             sharey='all')

            n_samples = self.validation_data[0].shape[0]

            self.indices = np.random.choice(np.arange(n_samples),
                                            size=self.n_choices,
                                            replace=False)

            x = self.validation_data[0][[self.indices]]

            for i, ax in enumerate(self.ax.flatten()):
                ax.clear()
                ax.imshow(x[i])

            plt.show()

        y_true = self.y_true[self.indices]
        y_pred = self.y_pred[self.indices]

        # check to see if masks, or labels
        try:
            n_imgs, img_height, img_width, img_channel = y_true.shape
            masks = np.concatenate(y_pred, y_true)
            labels = None
        except ValueError:
            n_imgs, n_classes = y_true.shape
            labels = (y_pred, y_true)
            masks = None

        for i, ax in enumerate(self.ax.flatten()):

            if masks is not None:
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, axis=2)

                for j in range(masks.shape[2]):
                    mask = masks[:, :, j]
                    if mask.max() > 0:
                        ax.contour(mask, [127.5, ],
                                   colors=self.mask_colors[j])

            if labels is not None:
                y_pred_i = labels[0][i].argmax()
                y_true_i = labels[1][i].argmax()
                ax.set_title('%s/%s' % (y_pred_i, y_true_i))
                if y_pred_i != y_true_i:
                    color = 'red' if y_true_i == 0 else 'magenta'
                else:
                    color = 'green'

                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(2.0)
                    ax.spines[axis].set_color(color)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

        # plt.subplots_adjust(wspace=0, hspace=0)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(3)


def config_cls_callbacks(run_name=None):
    callbacks = [
        ValidationPrediction(show_confusion_matrix=True),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.25,
                          patience=2,
                          verbose=1,
                          mode='auto',
                          min_lr=1e-7)
    ]
    if run_name:
        callbacks.extend([
            ModelCheckpoint(get_weights_filename(run_name),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=True),
            CSVLogger(filename=get_csv_filename(run_name))
        ])
    return callbacks


def config_seg_callbacks(run_name=None):
    callbacks = [
        ValidationPrediction(show_confusion_matrix=False),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=2,
                          verbose=1,
                          mode='auto',
                          min_lr=1e-7),
    ]
    if run_name:
        callbacks.extend([
            ModelCheckpoint(get_weights_filename(run_name),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=True),
            CSVLogger(filename=get_csv_filename(run_name))
        ])
    return callbacks
