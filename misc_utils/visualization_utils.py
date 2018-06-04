import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datasets.ISIC2018 import class_names
from matplotlib import colors


def plot_mask(axis_in, img_in, mask_in, title_in):
    mask_colors = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    img = img_in.copy()
    axis_in.clear()
    if mask_in.shape[2] > 1:
        mask_max = np.argmax(mask_in, axis=2)
        for mask_idx in range(1, mask_in.shape[2]):
            img[mask_max == mask_idx, :] = np.round(
                np.asarray(colors.colorConverter.to_rgb(mask_colors[mask_idx])) * 255)
    else:
        img[mask_in[:, :, 0] > 0.5, :] = np.round(np.asarray(colors.colorConverter.to_rgb('y')) * 255)

    axis_in.imshow(img)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    font_weight = 'bold'
    font_color = 'k'

    fontdict = {'family': 'serif',
                'color': font_color,
                'weight': font_weight,
                'size': 14,
                }

    # place a text box in upper left in axes coords
    axis_in.text(0.35, 0.95, title_in, transform=axis_in.transAxes, fontdict=fontdict, verticalalignment='top',
                 bbox=props)


class BatchVisualization(object):
    def __init__(self, images,
                 true_masks=None, pred_masks=None,
                 true_labels=None, pred_labels=None,
                 legends=None, **fig_kwargs):

        self.images = images
        self.n_images = self.images.shape[0]
        self.true_masks = true_masks
        self.pred_masks = pred_masks
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.legends = legends
        if true_labels is not None:
            self.ncols = fig_kwargs.get('ncols', 3)
            self.nrows = fig_kwargs.get('nrows', 3)
        else:  # Mask predictions
            self.ncols = fig_kwargs.get('ncols', 1)
            self.nrows = fig_kwargs.get('nrows', 3)

        self.batch_size = self.ncols * self.nrows
        self.start_idx = 0
        self.num_plots_per_image = 1

        if self.true_masks is not None:
            self.num_plots_per_image += 1

        if self.pred_masks is not None:
            self.num_plots_per_image += 1

        if self.true_masks is not None and len(self.true_masks.shape) == 3:
            self.true_masks = np.expand_dims(self.true_masks, axis=3)

        if self.pred_masks is not None and len(self.pred_masks.shape) == 3:
            self.pred_masks = np.expand_dims(self.pred_masks, axis=3)

        self.mask_type = fig_kwargs.get('mask_type', 'contour')

    def __call__(self, *args, **kwargs):

        self.fig, self.ax = plt.subplots(figsize=(10, 8),
                                         nrows=self.nrows,
                                         ncols=self.num_plots_per_image * self.ncols,
                                         sharex='all',
                                         sharey='all',
                                         gridspec_kw={'hspace': 0.,
                                                      'wspace': 0.
                                                      }
                                         )

        self.ax = np.ravel(self.ax)

        for ax in self.ax:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.axis('off')

        self.fig.subplots_adjust(bottom=0.2, hspace=0, wspace=0)

        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

        self.start_idx = 0

        self.update_batch()
        self.update_buttons()
        plt.show(block=True)

    def next(self, event):
        self.start_idx += self.batch_size
        self.update_batch()
        self.update_buttons()
        plt.show(block=True)

    def prev(self, event):
        self.start_idx -= self.batch_size
        self.update_batch()
        self.update_buttons()
        plt.show(block=True)

    def update_buttons(self):
        if self.start_idx + self.batch_size < self.n_images:
            self.axnext.set_visible(True)
            self.bnext.set_active(True)
        else:
            self.axnext.set_visible(False)
            self.bnext.set_active(False)

        if self.start_idx - self.batch_size >= 0:
            self.bprev.set_active(True)
            self.axprev.set_visible(True)
        else:
            self.bprev.set_active(False)
            self.axprev.set_visible(False)

    def update_batch(self):

        for ax_idx, image_idx in enumerate(range(self.start_idx,
                                                 min(self.start_idx + self.batch_size,
                                                     self.n_images))):

            img_ax_idx = ax_idx * self.num_plots_per_image

            self.ax[img_ax_idx].clear()
            self.ax[img_ax_idx].imshow(self.images[image_idx])

            if self.true_masks is not None:
                img_ax_idx += 1

                plot_mask(axis_in=self.ax[img_ax_idx],
                          img_in=self.images[image_idx],
                          mask_in=self.true_masks[image_idx],
                          title_in='GT mask')

            if self.true_labels is not None or self.pred_labels is not None:

                true_label = None if self.true_labels is None else class_names[np.argmax(self.true_labels[image_idx])]
                pred_label = None if self.pred_labels is None else class_names[np.argmax(self.pred_labels[image_idx])]

                if true_label is not None and pred_label is not None:
                    label = '%s -> %s' % (true_label, pred_label)
                    font_color = 'darkgreen' if true_label == pred_label else 'darkred'
                else:
                    label = '%s' % true_label if pred_label is None else pred_label
                    font_color = 'k'

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                font_weight = 'bold'

                fontdict = {'family': 'serif',
                            'color': font_color,
                            'weight': font_weight,
                            'size': 14,
                            }

                # place a text box in upper left in axes coords
                ax = self.ax[img_ax_idx]
                ax.text(0.35, 0.95, label, transform=ax.transAxes, fontdict=fontdict, verticalalignment='top',
                        bbox=props)

            if self.pred_masks is not None:
                img_ax_idx += 1

                plot_mask(axis_in=self.ax[img_ax_idx],
                          img_in=self.images[image_idx],
                          mask_in=self.pred_masks[image_idx],
                          title_in='Pred mask')

        if self.legends:
            plt.figlegend()

        for ax in self.ax:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.axis('off')

        plt.draw()


def view_by_batch(images, masks=None):
    batch_visualization = BatchVisualization(images, masks)
    batch_visualization()


if __name__ == "__main__":

    from datasets.ISIC2018 import load_training_data

    k_fold = 0

    for task_idx in range(1, 4):

        (x_train, y_train), _, _ = load_training_data(task_idx=task_idx,
                                                      output_size=224,
                                                      idx_partition=k_fold)

        x_train = x_train[:64]
        y_train = y_train[:64]

        if task_idx == 3:
            bv = BatchVisualization(images=x_train, true_labels=y_train)
            bv()
        else:
            bv = BatchVisualization(images=x_train, true_masks=y_train)
            bv()

    plot_debug = True

    if plot_debug is False:
        exit(0)

    # below for plot debugging purposes only
    for task_idx in range(1, 4):

        (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=task_idx,
                                                                       output_size=224,
                                                                       idx_partition=k_fold)

        x_train = x_train[:64]
        y_train = y_train[:64]
        y_valid = y_valid[:64]

        if task_idx == 3:
            bv = BatchVisualization(images=x_train, true_labels=y_train, pred_labels=y_valid)
            bv()
        else:
            bv = BatchVisualization(images=x_train, true_masks=y_train, pred_masks=y_valid)
            bv()
