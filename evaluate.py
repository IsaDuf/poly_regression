import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def report_loss(loss, config, mr):
    """Prints loss of model.

    Prints the loss of the model after training.

    If model is trained using cross-validation, the loss for each fold along with the mean loss over the k-fold are
    printed.

    If model is an ensemble or piecewise model (multiple parallel components), loss of each component (called cuts)
    along with the mean loss over the k-cuts are printed.
    printed.

    Parameters
    ----------
    loss : ndarray
        loss calculated during testing
    config : config

    Returns
    -------
    None
    """

    if config.cv:
        print("Loss on folds")

        for i, f_loss in enumerate(np.split(loss, config.kFold)):
            print("fold {} : {}".format(i, f_loss))

        print("Mean Loss {}\n".format(np.mean(loss)))

    elif mr:
        for i, r_loss in enumerate(np.split(loss, config.n_run)):
            print("Loss run {} : {}  | mean Loss {}".format(i, r_loss, np.mean(r_loss)))

        print("Mean Loss over runs {}".format(np.mean(loss)))
        print("STD Loss: {}\n".format(np.std(loss)))

    elif loss.shape != ():
        print("Loss on cuts {}".format(loss))
        print("Mean Loss {}\n".format(np.mean(loss)))

    else:
        print("Loss {}\n".format(loss))


def report_accuracy(acc, config, mr):
    """Prints final accuracy of model.

     Prints final accuracy of model based on `config.num_class` if the targets are discrete, or `config.num_map_class`
     if the targets are continuous.

    Parameters
    ----------
    acc : ndarray
        the accuracy calculated during testing
    config : config

    Returns
    -------
    None
    """

    if config.cv:
        print("Accuracy {}".format(acc))
        print("KFold Accuracy {}\n".format(np.mean(acc)))

    elif mr:
        print("Accuracy {}".format(acc))
        print("Mean Accuracy {}".format(np.mean(acc)))
        print("STD Accuracy: {}\n".format(np.std(acc)))

    else:
        print("Accuracy {}\n".format(acc))


def report_precision(label, pred, config, mr):
    """Prints a precision performance report of the model based on the predictions and labels (tags) distance.

    First, calculates the circular distance between the predictions and the continuous targets, then prints a report on
    the percentage of the predictions with errors smaller than diverse thresholds.

    The distance thresholds are calculated based on the range of the the circular continuous targets and the number of
    tags used over the entire range. For example, if the range of the targets is `[0, 359] and the number of tags is
    `360`, then the thresholds would be 1. The reports groups all the predictions that were at most `t` tags distance
    away from the prediction:
    distance of 0 tag: % of circular distance between the predictions and the continuous targets < 0.5
    distance of 1 tag: % of circular distance between the predictions and the continuous targets < 1.5
    distance of 2 tag: % of circular distance between the predictions and the continuous targets < 2.5

    Parameters
    ----------
    label : ndarray
        targets
    pred : ndarray
        logits
    config : config

    Returns
    -------
    None

    """
    print("\n====== Evaluating Precision of Regression Predictions ==========")
    label_range = config.max_range - config.min_range
    bound_big = label_range / 2
    bound_small = -label_range / 2

    # TODO preferred solution but memory explodes... investigate
    # error = label - pred
    # np.where(error > bound_big, (label - label_range) - pred, error)
    # np.where(error < bound_small, (label + label_range) - pred, error)
    # error = abs(error)
    # for i in range(50):
    #     print(error[i])

    error = np.zeros_like(label)
    for i in range(len(label)):
        error[i] = label[i] - pred[i]
        dist = error[i]
        if dist > bound_big:
            dist = (label[i] - label_range) - pred[i]
        elif dist < bound_small:
            dist = (label[i] + label_range) - pred[i]

        error[i] = abs(dist)

    if mr:
        per_run_error = np.split(error, config.n_run)

    td = label_range / config.num_tag
    for i in range(9):
        if mr:
            print("Mean tag distance of {}: {:.2f}".format(i, (error < (i * td + td / 2)).sum() / len(error) * 100))
            print("\tPer run: {}".format([round((run < (i * td + td / 2)).sum()
                                                          / len(run) * 100, 2) for run in per_run_error]))
        else:
            print("Tag distance of {}: {:.2f}".format(i, (error < (i * td + td / 2)).sum() / len(error) * 100))

    print()


def report_classification_results(label, pred, config):
    """Prints confusion matrix and classification reports.

    Prints confusion matrix and classification reports based on `config.num_class` if the targets are discrete, or
    `config.num_map_class` if the targets are continuous. Uses sklearn.metrics implementations.
    Parameters
    ----------
    label : ndarray
        targets
    pred : ndarray
        logits
    config : config

    Returns
    -------
    None

    """
    print("\n====== Evaluating as Classifier ==========")
    labels = None

    if config.num_class == 1:
        target_names = [str(i) for i in range(1, config.num_map_class + 1)]
    else:
        target_names = [str(i) for i in range(1, config.num_class + 1)]

    print(confusion_matrix(label, pred, labels=labels))
    print(classification_report(label, pred, labels=labels, target_names=target_names))
    
    
def evaluate_results(config, loss, accuracy, test_label, test_pred, cls_label, cls_pred, circ_label, circ_pred, mr=False):
    """Prints a complete report on the performance of a model including loss, accuracy, classification reports,
    confusion matrix and a precision report based on tag distance (except for SVM).

    Parameters
    ----------
    config : ndarray
        config
    loss : ndarray
        loss
    accuracy : ndarray
        accuracy
    test_label : ndarray
        targets
    test_pred : ndarray
        model's predictions
    cls_label : ndarray
        discrete target (class labels)
    cls_pred : ndarray
        model's discrete predictions
    circ_label : ndarray
        circular continuous target
    circ_pred : ndarray
        model's circular continuous predictions
    mr : bool, optional
        set to True if the report is for k_fold results

    Returns
    -------
    None
    """

    print("\n====================================\n====================================")
    print("\n--------TESTING RESULTS")
    report_loss(loss, config, mr)
    report_accuracy(accuracy, config, mr)

    if config.num_class > 1:
        report_classification_results(test_label, test_pred, config, mr)
    else:
        report_classification_results(cls_label, cls_pred, config, mr)

    if config.loss_type != 'svm':
        report_precision(circ_label, circ_pred, config, mr)


if __name__ == "__main__":

    print("Should implement tests")
