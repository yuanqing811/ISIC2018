
if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from skimage.transform import resize as sk_resize

    from paths import submission_dir, mkdir_if_not_exist
    from models import backbone
    from runs.seg_eval import task1_post_process
    from datasets.ISIC2018 import load_validation_data, load_test_data
    from misc_utils.prediction_utils import inv_sigmoid, sigmoid, cyclic_pooling, cyclic_stacking


    def task1_tta_predict(model, img_arr):
        img_arr_tta = cyclic_stacking(img_arr)
        mask_arr_tta = []
        for _img_crops in img_arr_tta:
            _mask_crops = model.predict(_img_crops)
            mask_arr_tta.append(_mask_crops)

        mask_crops_pred = cyclic_pooling(*mask_arr_tta)

        return mask_crops_pred

    backbone_name = 'vgg16'
    version = '0'
    task_idx = 1
    use_tta = False

    pred_set = 'validation'  # or test
    load_func = load_validation_data if pred_set == 'validation' else load_test_data
    images, image_names, image_sizes = load_func(task_idx=1, output_size=224)

    # max_num_images = 10
    max_num_images = images.shape[0]
    images = images[:max_num_images]
    image_names = image_names[:max_num_images]
    image_sizes = image_sizes[:max_num_images]

    y_pred = np.zeros(shape=(max_num_images, 224, 224))

    num_folds = 5

    print('Starting prediction for set %s with TTA set to %r with num_folds %d' % (pred_set, use_tta, num_folds))

    for k_fold in range(num_folds):
        print('Processing fold ', k_fold)
        model_name = 'task%d_%s' % (task_idx, backbone_name)
        run_name = 'task%d_%s_k%d_v%s' % (task_idx, backbone_name, k_fold, version)
        model = backbone(backbone_name).segmentation_model(load_from=run_name)
        if use_tta:
            y_pred += inv_sigmoid(task1_tta_predict(model=model, img_arr=images))[:, :, :, 0]
        else:
            y_pred += inv_sigmoid(model.predict(images))[:, :, :, 0]

    print('Done predicting -- now doing post-processing')

    y_pred = y_pred / num_folds
    y_pred = sigmoid(y_pred)

    y_pred = task1_post_process(y_prediction=y_pred, threshold=0.5, gauss_sigma=2.)

    output_dir = submission_dir + '/task1_' + pred_set
    mkdir_if_not_exist([output_dir])

    for i_image, i_name in enumerate(image_names):

        current_pred = y_pred[i_image]
        current_pred = current_pred * 255

        resized_pred = sk_resize(current_pred,
                                 output_shape=image_sizes[i_image],
                                 preserve_range=True,
                                 mode='reflect',
                                 anti_aliasing=True)

        resized_pred[resized_pred > 128] = 255
        resized_pred[resized_pred <= 128] = 0

        im = Image.fromarray(resized_pred.astype(np.uint8))
        im.save(output_dir + '/' + i_name + '_segmentation.png')

