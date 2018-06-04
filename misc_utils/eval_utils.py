import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils import compute_class_weight as sk_compute_class_weight

from datasets.ISIC2018 import classes, class_names
from misc_utils.print_utils import print_confusion_matrix, print_precision_recall


def get_confusion_matrix(y_true, y_pred, norm_cm=True, print_cm=True):
    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)

    cnf_mat = confusion_matrix(true_class, pred_class, labels=classes)

    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat/(cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    if print_cm:
        print_confusion_matrix(cm=total_cnf_mat, labels=class_names + ['TOTAL', ])

    return cnf_mat


def get_precision_recall(y_true, y_pred, print_pr=True):

    true_class = np.argmax(y_true, axis=1)
    pred_class = np.argmax(y_pred, axis=1)
    precision, recall, _, _ = precision_recall_fscore_support(y_true=true_class,
                                                              y_pred=pred_class,
                                                              labels=classes,
                                                              warn_for=())
    if print_pr:
        print_precision_recall(precision=precision, recall=recall, labels=class_names)

    return precision, recall


def compute_class_weights(y, wt_type='balanced', return_dict=True):
    # need to check if y is one hot
    if len(y.shape) > 1:
        y = y.argmax(axis=-1)

    assert wt_type in ['ones', 'balanced', 'balanced-sqrt'], 'Weight type not supported'

    classes = np.unique(y)
    class_weights = np.ones(shape=classes.shape[0])

    if wt_type == 'balanced' or wt_type == 'balanced-sqrt':

        class_weights = sk_compute_class_weight(class_weight='balanced',
                                                classes=classes,
                                                y=y)
        if wt_type == 'balanced-sqrt':
            class_weights = np.sqrt(class_weights)

    if return_dict:
        class_weights = dict([(i, w) for i, w in enumerate(class_weights)])

    return class_weights


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred) # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect))/(union - intersect +  1e-7)


def compute_jaccard(y_true, y_pred):

    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):

        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard/y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard/y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard
