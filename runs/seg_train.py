if __name__ == '__main__':
    from datasets.ISIC2018 import *
    from callback import config_seg_callbacks
    from misc_utils.print_utils import Tee, log_variable
    from misc_utils.filename_utils import get_log_filename
    from misc_utils.visualization_utils import BatchVisualization
    from keras.preprocessing.image import ImageDataGenerator
    from models import backbone
    import numpy as np
    import sys

    task_idx = 1
    version = '0'

    num_folds = 5

    for k_fold in range(num_folds):

        # backbone_name = 'unet'
        # backbone_name = 'inception_v3'
        # backbone_name = 'resnet50'
        # backbone_name = 'densenet169'

        backbone_name = 'vgg16'

        # Network architecture
        upsampling_type = 'deconv'
        bottleneck = True
        batch_normalization = False
        init_nb_filters = 32
        growth_rate = 2
        nb_blocks = 5
        nb_layers_per_block = 2
        max_nb_filters = 512

        encoder_activation = 'relu'
        decoder_activation = 'relu'
        use_activation = True
        use_soft_mask = False

        if backbone_name == 'unet':
            backbone_options = {
                'nb_blocks': nb_blocks,
                'init_nb_filters': init_nb_filters,
                'growth_rate': growth_rate,
                'nb_layers_per_block': nb_layers_per_block,
                'max_nb_filters': max_nb_filters,
                'activation': encoder_activation,
                'batch_normalization': batch_normalization,
            }
        else:
            backbone_options = {}

        # training parameter
        batch_size = 32
        initial_epoch = 0
        epochs = 25
        init_lr = 1e-4  # Note learning rate is very important to get this to train stably
        min_lr = 1e-7
        patience = 1

        # data augmentation parameters
        use_data_aug = True
        horizontal_flip = True
        vertical_flip = True
        rotation_angle = 180
        width_shift_range = 0.1
        height_shift_range = 0.1

        model_name = 'task%d_%s' % (task_idx, backbone_name)
        run_name = 'task%d_%s_k%d_v%s' % (task_idx, backbone_name, k_fold, version)
        from_run_name = None

        debug = False
        print_model_summary = True
        plot_model_summary = True

        logfile = open(get_log_filename(run_name=run_name), 'w+')
        original = sys.stdout
        sys.stdout = Tee(sys.stdout, logfile)

        assert task_idx in {1, 2}

        if task_idx == 1:
            metrics = ['jaccard_index',
                       'pixelwise_sensitivity',
                       'pixelwise_specificity']
        else:
            metrics = ['jaccard_index0',
                       'jaccard_index1',
                       'jaccard_index2',
                       'jaccard_index3',
                       'jaccard_index4',
                       'jaccard_index5']

        (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=task_idx,
                                                                       output_size=224,
                                                                       num_partitions=num_folds,
                                                                       idx_partition=k_fold)

        # Target should be of the type N x 224 x 224 x 1
        if len(y_train.shape) == 3:

            y_train = y_train[..., None]
            y_valid = y_valid[..., None]

        if y_train[0].max() > 1:
            if use_soft_mask:
                y_train = y_train / 255.
                y_valid = y_valid / 255.
            else:
                y_train = (y_train > 127.5).astype(np.uint8)
                y_valid = (y_valid > 127.5).astype(np.uint8)
        else:
            y_train = y_train.astype(np.uint8)
            y_valid = y_valid.astype(np.uint8)

        n_samples_train = x_train.shape[0]
        n_samples_valid = x_valid.shape[0]

        debug_visualize = False

        if debug_visualize:

            x_train = x_train[:32]
            y_train = y_train[:32]

            x_valid = x_valid[:32]
            y_valid = y_valid[:32]

            bv = BatchVisualization(images=x_train, true_masks=y_train)
            bv()

        callbacks = config_seg_callbacks(run_name)

        if from_run_name:
            model = backbone(backbone_name).segmentation_model(load_from=from_run_name, lr=init_lr)
        else:
            model = backbone(backbone_name, **backbone_options).segmentation_model(input_shape=x_train.shape[1:],
                                                                                   num_classes=y_train.shape[3],
                                                                                   upsampling_type=upsampling_type,
                                                                                   bottleneck=bottleneck,
                                                                                   init_nb_filters=init_nb_filters,
                                                                                   growth_rate=growth_rate,
                                                                                   nb_layers_per_block=nb_layers_per_block,
                                                                                   max_nb_filters=max_nb_filters,
                                                                                   activation=decoder_activation,
                                                                                   use_activation=use_activation,
                                                                                   save_to=run_name,
                                                                                   print_model_summary=print_model_summary,
                                                                                   plot_model_summary=plot_model_summary,
                                                                                   lr=init_lr,
                                                                                   loss='ce',
                                                                                   metrics=metrics,
                                                                                   name=model_name)

        log_variable(var_name='input_shape', var_value=x_train.shape[1:])
        log_variable(var_name='num_classes', var_value=y_train.shape[3])
        log_variable(var_name='upsampling_type', var_value=upsampling_type)
        log_variable(var_name='bottleneck', var_value=bottleneck)
        log_variable(var_name='init_nb_filters', var_value=init_nb_filters)
        log_variable(var_name='growth_rate', var_value=growth_rate)
        log_variable(var_name='nb_layers_per_block', var_value=nb_layers_per_block)
        log_variable(var_name='max_nb_filters', var_value=max_nb_filters)
        log_variable(var_name='encoder_activation', var_value=encoder_activation)
        log_variable(var_name='decoder_activation', var_value=decoder_activation)
        log_variable(var_name='batch_normalization', var_value=batch_normalization)
        log_variable(var_name='use_activation', var_value=use_activation)
        log_variable(var_name='use_soft_mask', var_value=use_soft_mask)

        log_variable(var_name='batch_size', var_value=batch_size)
        log_variable(var_name='initial_epoch', var_value=initial_epoch)
        log_variable(var_name='epochs', var_value=epochs)
        log_variable(var_name='init_lr', var_value=init_lr)
        log_variable(var_name='min_lr', var_value=min_lr)
        log_variable(var_name='patience', var_value=patience)

        log_variable(var_name='use_data_aug', var_value=use_data_aug)

        if use_data_aug:

            log_variable(var_name='horizontal_flip', var_value=horizontal_flip)
            log_variable(var_name='vertical_flip', var_value=vertical_flip)
            log_variable(var_name='width_shift_range', var_value=width_shift_range)
            log_variable(var_name='height_shift_range', var_value=height_shift_range)
            log_variable(var_name='rotation_angle', var_value=rotation_angle)

        log_variable(var_name='n_samples_train', var_value=n_samples_train)
        log_variable(var_name='n_samples_valid', var_value=n_samples_valid)

        sys.stdout.flush()  # need to make sure everything gets written to file

        if use_data_aug:

            data_gen_args = dict(horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 rotation_range=rotation_angle,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range)

            image_datagen = ImageDataGenerator(**data_gen_args)
            mask_datagen = ImageDataGenerator(**data_gen_args)

            seed = 42

            image_datagen.fit(x_train, augment=True, seed=seed)
            mask_datagen.fit(y_train, augment=True, seed=seed)

            image_generator = image_datagen.flow(x=x_train, batch_size=batch_size, seed=seed)
            mask_generator = mask_datagen.flow(x=y_train, batch_size=batch_size, seed=seed)

            train_generator = zip(image_generator, mask_generator)

            model.fit_generator(generator=train_generator,
                                steps_per_epoch=n_samples_train // batch_size,
                                epochs=epochs,
                                initial_epoch=initial_epoch,
                                verbose=1,
                                validation_data=(x_valid, y_valid),
                                callbacks=callbacks,
                                workers=8,
                                use_multiprocessing=False)
        else:

            model.fit(x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_valid, y_valid),
                      shuffle=True,
                      callbacks=callbacks)

        sys.stdout = original
