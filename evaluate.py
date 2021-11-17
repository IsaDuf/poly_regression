import os
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import h5py


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


def report_classification_results(label, pred, config, cycle=None):
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
    if cycle is not None:
        num_map = {"hour": 24, "day": 7, "week": 52}

    if config.num_class == 1:
        if config.label_type == "tricycle":
            num_map_class = num_map[cycle]
            target_names = [str(i) for i in range(1, num_map_class + 1)]
            labels = [i for i in range(0, num_map_class)]
        else:
            target_names = [str(i) for i in range(1, config.num_map_class + 1)]
    else:
        target_names = [str(i) for i in range(1, config.num_class + 1)]

    print(confusion_matrix(label, pred, labels=labels))
    print(classification_report(label, pred, labels=labels, target_names=target_names))


def get_va_scores(config, r_label, r_pred, val_label, val_pred, arous_label, arous_pred, mr):

    if mr:
        val_pred_per_run = np.split(val_pred, config.n_run)
        arous_pred_per_run = np.split(arous_pred, config.n_run)
        val_label_per_run = np.split(val_label, config.n_run)
        arous_label_per_run = np.split(arous_label, config.n_run)

        med = []
        rmse_val = []
        rmse_arous = []
        r2_val = []
        r2_arous = []
        for i in range(config.n_run):
            med.append(np.mean(np.sqrt(np.square(val_pred_per_run[i] - val_label_per_run[i]) +
                                  np.square(arous_pred_per_run[i] - arous_label_per_run[i]))))

            rmse_val.append(np.sqrt(np.mean((val_pred_per_run[i] - val_label_per_run[i]) ** 2)))
            rmse_arous.append(np.sqrt(np.mean((arous_pred_per_run[i] - arous_label_per_run[i]) ** 2)))

            r2_val.append(1 - np.sum((val_label_per_run[i] - val_pred_per_run[i]) ** 2) /
                          np.sum((val_label_per_run[i] - np.mean(val_label_per_run[i])) ** 2))
            r2_arous.append(1 - np.sum((arous_label_per_run[i] - arous_pred_per_run[i]) ** 2) /
                            np.sum((arous_label_per_run[i] - np.mean(arous_label_per_run[i])) ** 2))

        print("\nMean Euclidean Distance:", med)
        print("MED over run:", np.mean(med))

        print("\nVal RMSE:", rmse_val)
        print("Val RMSE over run:", np.mean(rmse_val))

        print("\nArous RMSE:", rmse_arous)
        print("Arous RMSE over run:", np.mean(rmse_arous))

        print("\nR^2 Val :", r2_val)
        print("R^2 Val over run:", np.mean(r2_val))

        print("\nR^2 Arous:", r2_arous)
        print("R^2 Arous over run:", np.mean(r2_arous))

    else:
        # get MED
        med = np.mean(np.sqrt(np.square(val_pred - val_label) + np.square(arous_pred - arous_label)))
        print("\nMean Euclidean Distance:", med)

        # get RMSE
        rmse_val = np.sqrt(np.mean((val_pred - val_label) ** 2))
        rmse_arous = np.sqrt(np.mean((arous_pred - arous_label) ** 2))
        print("\nRMSE Valence:", rmse_val)
        print("RMSE Arousal:", rmse_arous)

        # get R2
        r2_val = 1 - np.sum((val_label - val_pred) ** 2) / np.sum((val_label - np.mean(val_label)) ** 2)
        r2_arous = 1 - np.sum((arous_label - arous_pred) ** 2) / np.sum((arous_label - np.mean(arous_label)) ** 2)
        print("\nR^2 Valence:", r2_val)
        print("R^2 Arousal:", r2_arous)

    return med, r2_val, r2_arous, rmse_val, rmse_arous
    
    
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

    elif config.label_type == "tricycle":
        mean_mad = np.mean(mad, axis=0)
        median_mad = np.median(mad, axis=0)
        cycles = ["hour", "day", "week"]
        for i, cycle in enumerate(cycles):
            print("\n-------- Report for {}".format(cycle))
            report_classification_results(cls_label[:, i], cls_pred[:, i], config, cycle)
            print("mean angular displacement: ", mean_mad[i])
            print("median angular displacement: ", median_mad[i])
            print()

    else:
        mean_mad = np.mean(mad)
        median_mad = np.median(mad)
        if mr:
            mad_run = np.split(mad, config.n_run)
            mean_mad_run = np.mean(mad_run, 1)
            median_mad_run = np.median(mad_run, 1)
            print("mean ad: ", mean_mad_run)
            print("mean ad over runs: ", np.mean(mean_mad_run))

            print("mad: ", median_mad_run)
            print("mad over runs: ", np.median(median_mad_run))

        else:
            print("mean angular displacement: ", mean_mad)
            print("median angular displacement: ", median_mad)

        if config.label_type == "va":
            get_va_scores(config, r_label, r_pred, val_label, val_pred, arous_label, arous_pred, mr)
        report_classification_results(cls_label, cls_pred, config)

    if config.loss_type != 'svm' and config.label_type != "tricycle":
        report_precision(circ_label, circ_pred, config, mr)


def save_results_to_h5(config, mean_mad, median_mad, med=None, r2_val=None, r2_arous=None, rmse_val=None, rmse_arous=None):
    """Save model's accuracy to <results_dir>/results.h5

    The model types (automatically expended by path_util and corresponding to the string appended to logs and save
    directories and found in config.model_type). are the keys for the accuracy entry.
    <results_directory> is automatically set to correspond to config.save_dir + dataset. This means the results of all
    models trained on a given dataset are in <results_dir>/results.h5.

    If the accuracy of a model is already in results.h5, the old value is replaced with the new value.

    Parameters
    ----------
    config : config
    acc : ndarray
        accuracy

    Returns
    -------
    None
    """
    print("\n\n ========== Saving to h5 ==========")
    results = list(filter(None, [np.mean(mean_mad), np.mean(median_mad), med, r2_val, r2_arous, rmse_val, rmse_arous]))
    results_name = ['mean_mad', 'median_mad', 'med', 'r2_val', 'r2_arous', 'rmse_val', 'rmse_arous']
    # save results to h5 file
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    done = False
    if not config.eq1_pieces:
        for i, result in enumerate(results):
            print("Saving {} results to results.h5".format(results_name[i]))
            # TODO add loss
            while not done:
                try:
                    f = h5py.File(config.results_dir + "/results.h5", "a")

                    try:
                        saved = f["{}/{}".format(results_name[i], config.model_type)]
                        print("Results already saved, replacing with new value")
                        try:
                            saved[...] = result
                        except TypeError:
                            print('problem saving data {}, exiting without saving'.format(results_name[i]))

                    except KeyError:
                        f.create_dataset("{}/{}".format(results_name[i], config.model_type), data=result)

                    f.close()
                    done = True

                except:
                    time.sleep(1)
                    print("File is used by another process, trying again in 1 sec")

            done = False


if __name__ == "__main__":

    print("Should implement tests")
