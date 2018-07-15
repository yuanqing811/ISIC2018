
from scipy.ndimage.filters import gaussian_filter
from datasets.ISIC2018 import *
from models import backbone
from misc_utils.visualization_utils import BatchVisualization
from misc_utils.eval_utils import compute_jaccard
from skimage.morphology import label as sk_label


def task1_post_process(y_prediction, threshold=0.5, gauss_sigma=0.):

    for im_index in range(y_prediction.shape[0]):

        # smooth image by Gaussian filtering

        if gauss_sigma > 0.:
            y_prediction[im_index] = gaussian_filter(input=y_prediction[im_index], sigma=gauss_sigma)

        thresholded_image = y_prediction[im_index] > threshold

        # find largest connected component

        labels, num_labels = sk_label(thresholded_image, return_num=True)

        max_label_idx = -1
        max_size = 0

        for label_idx in range(0, num_labels + 1):

            if np.sum(thresholded_image[labels == label_idx]) == 0:
                continue

            current_size = np.sum(labels == label_idx)

            if current_size > max_size:
                max_size = current_size
                max_label_idx = label_idx

        if max_label_idx > -1:
            y_prediction[im_index] = labels == max_label_idx
        else: # no predicted pixels found
            y_prediction[im_index] = y_prediction[im_index] * 0

    return y_prediction


if __name__ == '__main__':

    backbone_name = 'vgg16'
    # backbone_name = 'inception_v3'
    k_fold = 0
    version = '0'
    task_idx = 1

    model_name = 'task%d_%s' % (task_idx, backbone_name)
    run_name = 'task%d_%s_k%d_v%s' % (task_idx, backbone_name, k_fold, version)

    _, (x, y_true), _ = load_training_data(task_idx=task_idx, output_size=224)

    if len(y_true.shape) == 3:
        y_true = y_true[..., None]

    if y_true[0].max() > 1:
        y_true = (y_true > 127.5).astype(np.uint8)

    model = backbone(backbone_name).segmentation_model(load_from=run_name, lr=0.001)

    # max_num_images = 32
    max_num_images = x.shape[0]
    x = x[:max_num_images]
    y_true = y_true[:max_num_images]

    y_pred = model.predict(x, batch_size=8)

    if task_idx == 1:
        y_pred = task1_post_process(y_prediction=y_pred, threshold=0.5, gauss_sigma=2.)
        mean_jaccard, thresholded_jaccard = compute_jaccard(y_true=y_true, y_pred=y_pred)
        print('Mean jaccard = %.3f, Thresholded Jaccard = %.3f ' % (mean_jaccard, thresholded_jaccard))

    bv = BatchVisualization(images=x,
                            true_masks=y_true,
                            pred_masks=y_pred)
    bv()

    # scores = model.evaluate(x, y_true, batch_size=32, verbose=1)
    # print(scores)
