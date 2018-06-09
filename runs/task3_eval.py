if __name__ == '__main__':

    from datasets.ISIC2018 import *
    from models import backbone
    from misc_utils.visualization_utils import BatchVisualization
    from misc_utils.eval_utils import get_confusion_matrix, get_precision_recall

    backbone_name = 'inception_v3'
    k_fold = 0
    version = '0'
    run_name = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version

    _, (x, y_true), _ = load_training_data(task_idx=3,
                                           output_size=224,
                                           idx_partition=k_fold)

    model = backbone(backbone_name).classification_model(load_from=run_name)

    # max_num_images = 32
    max_num_images = x.shape[0]
    x = x[:max_num_images]
    y_true = y_true[:max_num_images]

    y_pred = model.predict(x)

    _ = get_confusion_matrix(y_true=y_true, y_pred=y_pred, print_cm=True)
    get_precision_recall(y_true=y_true, y_pred=y_pred)

    bv = BatchVisualization(images=x,
                            true_labels=y_true,
                            pred_labels=y_pred)
    bv()
