import os
import numpy as np
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from datasets.ISIC2018 import task12_image_ids
from datasets.ISIC2018 import TASK2_ATTRIBUTE_NAMES, task2_image_ids
from datasets.ISIC2018 import get_task12_resized_img_dir
from datasets.ISIC2018 import get_task2_resized_gt_dir
from datasets.ISIC2018 import partition_indices
from keras import backend as K


def task12_image_filename_iterator(image_ids, target_size=None):
    dirname = get_task12_resized_img_dir(output_size=target_size)
    return [os.path.join(dirname, '%s.jpg' % image_id)
            for image_id in image_ids]


def task2_gt_filename_iterator(image_ids, attribute_name, target_size=None):
    dirname = get_task2_resized_gt_dir(output_size=target_size)
    return [os.path.join(dirname, '%s_attribute_%s.png' % (image_id, attribute_name))
            for image_id in image_ids]


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: None or {'input', 'binary', 'categorical'}, default: 'categorical'
            Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, filenames, image_data_generator,
                 target_size=1024, color_mode='rgb',
                 classes=None, class_mode=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 interpolation='nearest'):

        if data_format is None:
            data_format = K.image_data_format()

        self.filenames = filenames
        self.image_data_generator = image_data_generator
        self.target_size = (target_size, target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.samples = len(filenames)
        self.filenames = filenames
        self.num_classes = 0

        if self.classes is not None:
            self.num_classes = len(self.classes)

        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            print(self.filenames[j])
            img = load_img(self.filenames[j], grayscale=grayscale)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class Task2DataGenerator(object):
    """
    a wrapper for keras ImageDataGenerator

    usage:         model.fit_generator(generator=train_generator,
                            steps_per_epoch=n_samples_train // batch_size,
                            epochs=epochs,
                            initial_epoch=initial_epoch,
                            verbose=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=False)
    """
    def __init__(self,
                 target_size=512,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 image_preprocessing_function=None,
                 mask_preprocessing_function=None,
                 attribute_names=None,
                 data_format=None,
                 num_partitions=5,
                 idx_partition=0,
                 verbose=False):

        self.verbose = verbose
        self.data_format = data_format if data_format else K.image_data_format()

        if attribute_names is None:
            self.attribute_names = TASK2_ATTRIBUTE_NAMES
            self.image_ids = task12_image_ids
        else:
            self.image_ids = []
            self.attribute_names = attribute_names
            for attribute_name in self.attribute_names:
                assert attribute_name in TASK2_ATTRIBUTE_NAMES
                attribute_image_ids = task2_image_ids[attribute_name]
                self.image_ids.extend(attribute_image_ids)

            if len(self.attribute_names) > 1:
                self.image_ids.sort()

        self.samples = len(self.image_ids)
        self.target_size = target_size
        train_indices, valid_indices, _ = partition_indices(self.samples,
                                                            k=num_partitions,
                                                            i=idx_partition,
                                                            test_split=0.)

        self.training_image_ids = [self.image_ids[i] for i, val in enumerate(train_indices) if val]
        self.validation_image_ids = [self.image_ids[i] for i, val in enumerate(valid_indices) if val]

        self.num_training_samples = len(self.training_image_ids)
        self.num_validation_samples = len(self.validation_image_ids)

        self.image_data_generator = ImageDataGenerator(
            preprocessing_function=image_preprocessing_function,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode='nearest')

        self.mask_data_generator = ImageDataGenerator(
            preprocessing_function=mask_preprocessing_function,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            fill_mode='constant', cval=0.0)

    def flow(self,
             batch_size=32,
             shuffle=True, seed=42,
             save_to_dir=None,
             save_prefix='',
             save_format='jpg',
             interpolation='nearest',
             subset=None):

        assert subset in {'validation', 'training', None}

        if subset is 'validation':
            image_ids = self.training_image_ids
        elif subset is 'training':
            image_ids = self.validation_image_ids
        else:
            image_ids = self.image_ids

        image_iterator = DirectoryIterator(
            task12_image_filename_iterator(image_ids, self.target_size),
            self.image_data_generator,
            target_size=self.target_size,
            class_mode=None,
            color_mode='rgb',
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            interpolation=interpolation)

        attr_iterators = []
        for attribute_name in self.attribute_names:
            attr_iterator = DirectoryIterator(
                task2_gt_filename_iterator(image_ids, attribute_name, self.target_size),
                self.mask_data_generator,
                target_size=self.target_size,
                class_mode=None,
                color_mode='grayscale',
                data_format=self.data_format,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                interpolation=interpolation
            )
            attr_iterators.append(attr_iterator)
        iterators = [image_iterator] + attr_iterators

        train_generator = zip(*iterators)

        for batch in train_generator:
            x_batch = batch[0]
            if len(batch[1:]) == 1:
                y_batch = batch[1]
            else:
                y_batch = np.concatenate(batch[1:], axis=-1)
            y_batch = (y_batch > 127.5).astype(np.float32)
            yield x_batch, y_batch


if __name__ == '__main__':
    print('testing Task2DataGenerator ...')
    from matplotlib import pyplot as plt

    plt.interactive(False)

    attribute_names = ['pigment_network', ]
    data_gen = Task2DataGenerator(attribute_names=attribute_names)
    count = 0
    colors = ['r', 'b', 'g', 'y', 'm']
    for x, y in data_gen.flow(batch_size=1):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(x[0].astype(np.uint8))
        if y.any():
            for i in range(y[0].shape[-1]):
                if y[0, :, :, i].max() > 0.:
                    c = ax.contour(y[0, :, :, i].astype(np.uint8),
                                   [127.5, ], colors=colors[i])
                    c.collections[0].set_label(attribute_names[i])
            plt.legend(loc='upper left')
            plt.show()
        # plt.savefig('test/fig%d' % count)
        count += 1
