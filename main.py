import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from math import ceil, sqrt
from tensorboardX import SummaryWriter

from config import get_config, print_usage
from models import Classifier, PWCirc
from utils import ptocirc, path_util, preprocess
import evaluate

from loader import CircMood
from circ_loss import CIRCLoss

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') if CUDA else torch.device('cpu')
print(DEVICE)
PIN_MEM = {'pin_memory': True} if CUDA else {}

RUN = 0


def data_criterion(config):
    """Loss objet - from config"""

    if config.loss_type == "cross_entropy":
        data_loss = nn.CrossEntropyLoss()
    elif config.loss_type == "svm":
        data_loss = nn.MultiMarginLoss()
    elif config.loss_type == "mse":
        data_loss = nn.MSELoss()
    elif config.loss_type == "circ_mse":
        data_loss = CIRCLoss(config)
    else:
        print("no loss function specified; Using default: cross-entropy")
        data_loss = nn.CrossEntropyLoss()

    return data_loss


def model_criterion(config):
    """L2 loss - Regularization"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param**2)

        return loss * config.l2_reg

    return model_loss


def dataset_init(config, data_type, dfile=None):
    """Init dataset: only dataset implemented is CircMood"""
    if 'circMood' in config.train:
        data = CircMood(config, data_type, dfile)

    else:
        raise NotImplementedError("This dataset is not implemented (see loader.py to implement new datasets")
    return data


def model_init(config, data):
    """Init model: PWCirc is ensemble model (polygonal approximation)
    Classifier can be used as a classifier (num_class > 1 ) or regression (num_class = 1)
    """
    if config.circ_sync:
        model = PWCirc(config=config, input_shape=data.data_shape)

    else:
        model = Classifier(config=config, input_shape=data.data_shape)

    return model


def bias_init(model, config):
    """Rotate biases for the ensemble model (circ_sync)"""
    print("rotating biases")
    data_range = config.max_range - config.min_range
    # stop = config.num_cuts
    for i, param in enumerate(model.multi_head_linear.bias):
        # if i < stop:
        mu = config.min_range + (data_range / 2) / config.num_cuts \
                     + i * (data_range / config.num_cuts)
        torch.nn.init.normal_(param, mu, 0.001)


def optim_init(config, model):
    if config.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=config.amsgrad)

    elif config.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, nesterov=config.nesterov)

    else:
        raise NotImplementedError()

    return optimizer


def train(config):
    """ Main training function """
    print(15 * "=")
    print("Training")
    print("\nStart training on device: {}".format(DEVICE))

    # only way to pass run number to workers init function
    global RUN
    RUN = config.run

    # Initialize Dataset
    train_data = dataset_init(config, "train", config.train)
    val_data = dataset_init(config, "val", train_data.dfile)

    print("Training fold {}".format(config.k))
    print("\nTraining on {} data samples ".format(len(train_data.tids)))
    print("Validation on {} data samples \n".format(len(val_data.tids)))
    print("Data shape is {} ".format(train_data.data_shape))
    print(config.mode)

    # select number of epoch based on num_it iterations
    bpere = ceil(len(train_data.tids)/config.bs)
    config.num_e = int(config.num_it/bpere)

    print("Number of epoch {}, Batch per epoch {}".format(config.num_e, bpere))
    print("lr : {}, l2reg : {}\n".format(config.lr, config.l2_reg))

    # Create model
    print("Model init seed: {}".format(1023210 + RUN))
    torch.manual_seed(1023210 + RUN)
    model = model_init(config, train_data)

    # Create data loader
    tr_data_loader = DataLoader(
        dataset=train_data,
        batch_size=config.bs,
        num_workers=4 if config.one_batch else 4,
        shuffle=True,
        drop_last=False,
        **PIN_MEM)

    va_data_loader = DataLoader(
        dataset=val_data,
        batch_size=config.bs,
        num_workers=0 if config.one_batch else 4,
        shuffle=False,
        drop_last=False,
        **PIN_MEM)

    print("Data loader and model created")

    # Move model to gpu if cuda available
    if torch.cuda.is_available():
        model = model.to(DEVICE)
        print("model moved to CUDA")

    print(model)

    # Set model for training
    model.train()
    print("\nmodel in training")

    # rotate bias
    if config.circ_sync and config.rotate_bias:
        bias_init(model=model, config=config)

    # define loss
    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    # create optimizer
    optimizer = optim_init(config, model)

    # Create summary writer and directories
    tr_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train/" + str(config.k)))
    va_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid/" + str(config.k)))

    # Create log directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    # Create save directory if it does not exist
    save_dir = config.save_dir + str(config.k)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize training
    data_range = config.max_range - config.min_range
    iter_idx = -1
    best_at = -1
    best_va_acc = -1
    best_va_loss = 100000
    max_cutoff = 50000
    min_diff = 25000
    checkpoint_file = os.path.join(save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(save_dir, "best_model.pth")

    board_stats = {"loss": None,
                   "acc": None,
                   "loss_on_data": None
                   }

    # Check for existing training results.
    if os.path.exists(checkpoint_file):
        if config.resume:
            print("Checkpoint found; Resuming")
            load_res = torch.load(checkpoint_file, map_location=DEVICE)
            iter_idx = load_res["iter_idx"]
            print("at iter_index {}".format(iter_idx))

            best_va_acc = load_res["best_va_acc"]

            try:
                print("found best at")
                load_best = torch.load(bestmodel_file)
                best_at = load_best["iter_idx"]
                print(best_at)
                load_best = None

            except:
                best_at = 0

            model.load_state_dict(load_res["model"])
            optimizer.load_state_dict(load_res["optimizer"])
            config.num_e = int((config.num_it - iter_idx) / bpere)
            print("remaining epoch: ", config.num_e)

        else:
            os.remove(checkpoint_file)

    # Training loop
    print("entering training loop")

    for epoch in range(config.num_e):
        prefix = "Training Epoch {:3d}: ".format(epoch)

        # use one batch to debug (should be able to memorize one batch)
        if config.one_batch:
            torch.manual_seed(0)

        # for data in tqdm(tr_data_loader):
        for data in tr_data_loader:
            iter_idx += 1

            if config.one_batch:
                torch.manual_seed(0)

            # Split the data
            x, y, y_class = data

            # Use class labels for classifiers
            if config.label_type == "class":
                y = y_class

            # Send data to GPU if available
            if torch.cuda.is_available():
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                y_class = y_class.to(DEVICE)

            # Zero gradients in the optimizer
            optimizer.zero_grad()

            # Forward pass
            logits = model.forward(x)

            # Compute gradients
            if config.circ_sync:
                y_cuts = []
                for cut in range(config.num_cuts):
                    y_cuts.append(y.clone())
                    y_cuts[cut] = preprocess.rotate_to_range(y_cuts[cut],
                                                             config.min_range + cut * (data_range / config.num_cuts),
                                                             config.max_range + cut * (data_range / config.num_cuts))

                y_cuts = torch.stack(y_cuts, dim=1)

                loss = data_loss(logits, y_cuts) + model_loss(model)

            else:
                if config.label_type == "class":
                    y = y.squeeze()

                loss = data_loss(logits, y) + model_loss(model)

            loss.backward()

            # Update parameters
            optimizer.step()

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # Compute accuracy
                with torch.no_grad():

                    # get accuracy
                    # for classifiers
                    if config.label_type == "class":
                        pred = torch.argmax(logits, dim=1)
                        acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                        board_stats["acc"] = acc

                    # for regressors
                    else:
                        if config.circ_sync:
                            pred = logits.detach().squeeze()
                            circ_pred = preprocess.circmean(pred,
                                                            high=config.max_range,
                                                            low=config.min_range,
                                                            dim=-1)

                        else:
                            circ_pred = logits.detach().squeeze()


                        cls_rcirc_pred = preprocess.label_to_class(circ_pred,
                                                                   config.num_map_class,
                                                                   config.min_range,
                                                                   config.max_range)

                        acc = torch.mean(torch.eq(cls_rcirc_pred, y_class).float()) * 100.0
                        board_stats["acc"] = acc

                # Write to tensorboard
                board_stats["loss"] = loss
                write_to_board(board_stats, tr_writer, iter_idx)

                # Save most recent model
                # if config.label_type == "class":
                torch.save({
                    "iter_idx": iter_idx,
                    "best_va_acc": best_va_acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint_file)

                # else:
                #     torch.save({
                #         "iter_idx": iter_idx,
                #         "best_va_loss": best_va_loss,
                #         "model": model.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #     }, checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                va_loss = []
                va_acc = []

                # Set model for evaluation
                model = model.eval()

                # va_iter = 0
                for data in va_data_loader:
                    if config.one_batch:
                        torch.manual_seed(0)

                    x, y, y_class = data

                    if config.label_type == "class":
                        y = y_class

                    if torch.cuda.is_available():
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        y_class = y_class.to(DEVICE)

                    with torch.no_grad():
                        # Compute logits
                        logits = model.forward(x)

                        # special case: ensemble
                        if config.circ_sync:
                            y_cuts = []

                            for cut in range(config.num_cuts):
                                y_cuts.append(y.clone())
                                y_cuts[cut] = preprocess.rotate_to_range(y_cuts[cut],
                                                                         config.min_range + cut * (
                                                                                         data_range / config.num_cuts),
                                                                         config.max_range + cut * (
                                                                                         data_range / config.num_cuts))

                            y_cuts = torch.stack(y_cuts, dim=1)

                            loss = data_loss(logits, y_cuts) + model_loss(model)
                            va_loss.append(loss.cpu().numpy())

                            # make class predictions and calculate accuracy
                            pred = logits.detach().squeeze()
                            circ_pred = preprocess.circmean(pred,
                                                            high=config.max_range,
                                                            low=config.min_range,
                                                            dim=-1)

                            cls_rcirc_pred = preprocess.label_to_class(circ_pred,
                                                                       config.num_map_class,
                                                                       config.min_range,
                                                                       config.max_range)

                            acc = torch.mean(torch.eq(cls_rcirc_pred, y_class).float()) * 100.0
                            va_acc += [acc.cpu().numpy()]

                        # classifiers
                        elif config.label_type == "class":
                            pred = torch.argmax(logits, dim=1)
                            acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                            loss = data_loss(logits, y.squeeze()) + model_loss(model)
                            va_acc += [acc.cpu().numpy()]
                            va_loss += [loss.detach().cpu().numpy()]

                        # all other continuous models
                        else:
                            # TODO get VA model in here
                            loss = data_loss(logits, y) + model_loss(model)
                            cls_rcirc_pred = 0
                            acc = 0
                            va_acc += [acc]
                            va_loss += [loss.detach().cpu().numpy()]

                # Set model back for training
                model = model.train()

                # Write to tensorboard
                va_loss = np.mean(va_loss)
                va_acc = np.mean(va_acc)
                board_stats["loss"] = np.mean(va_loss)
                board_stats["acc"] = np.mean(va_acc)

                write_to_board(board_stats, va_writer, iter_idx)

                # Replace best model if improved
                # if config.label_type == "class":
                if va_acc > best_va_acc:
                    best_va_acc = va_acc
                    best_at = iter_idx

                    torch.save({
                        "iter_idx": iter_idx,
                        "best_va_acc": best_va_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, bestmodel_file)

            if iter_idx >= config.num_it/config.bs:
                print("Training finished at it {}.".format(iter_idx))
                return


def test(config, model_to_load="best_model.pth"):
    print(15 * "=")
    print("Testing")

    config.resume = False

    # Initialize Dataset and loader
    test_data = dataset_init(config, "test", config.test)

    te_data_loader = DataLoader(
        dataset=test_data,
        batch_size=config.bs,
        num_workers=2,
        shuffle=False
    )

    # Create model
    model = model_init(config, test_data)

    print('\nmodel created')

    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    # Move to GPU if you have one.
    if torch.cuda.is_available():
        model = model.to(DEVICE)

    # Load our best model
    print(DEVICE)
    save_dir = config.save_dir + str(config.k)
    load_res = torch.load(os.path.join(save_dir, model_to_load), map_location=DEVICE)
    print("best model at iter_index {}".format(load_res["iter_idx"]))
    model.load_state_dict(load_res["model"])

    # Set model for testing
    model.eval()

    # The Test loop
    te_loss = []
    te_acc = []
    te_label = []
    te_pred = []
    te_cls_label = []
    te_cls_pred = []
    te_circ_label = []
    te_circ_pred = []

    data_range = config.max_range - config.min_range

    # for data in tqdm(te_data_loader, desc=prefix):
    iter_idx = -1
    for data in te_data_loader:
        iter_idx += 1

        x, y, y_class = data

        if config.label_type == "class":
            y_circ = y
            y = y_class

        # Send data to GPU if we have one
        if torch.cuda.is_available():
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_class = y_class.to(DEVICE)

            if config.label_type == "class":
                y_circ = y_circ.to(DEVICE)

        # Don't invoke gradient computation
        with torch.no_grad():
            # Compute logits
            logits = model.forward(x)

            # Special case ensemble model
            if config.circ_sync:
                y_cuts = []

                for cut in range(config.num_cuts):
                    y_cuts.append(y.clone())
                    y_cuts[cut] = preprocess.rotate_to_range(y_cuts[cut],
                                                             config.min_range + cut * (data_range / config.num_cuts),
                                                             config.max_range + cut * (data_range / config.num_cuts))

                y_cuts = torch.stack(y_cuts, dim=1)

                loss = data_loss(logits, y_cuts) + model_loss(model)
                te_loss.append(loss.cpu().numpy())

                # make class predictions and calculate accuracy
                pred = logits.detach().squeeze()

                circ_pred = preprocess.circmean(pred, high=config.max_range, low=config.min_range, dim=-1)

                cls_rcirc_pred = preprocess.label_to_class(circ_pred,
                                                           config.num_map_class,
                                                           config.min_range,
                                                           config.max_range)

                acc = torch.mean(torch.eq(cls_rcirc_pred, y_class).float()) * 100.0

                te_acc += [acc.cpu().numpy()]
                te_pred.extend(circ_pred.cpu().numpy())
                te_circ_pred.extend(circ_pred.cpu().numpy())
                te_cls_pred.extend(cls_rcirc_pred.cpu().numpy())

            # classifiers
            elif config.label_type == "class":
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean(torch.eq(pred, y).float()) * 100.0
                loss = data_loss(logits, y.squeeze())

                te_acc += [acc.cpu().numpy()]
                te_loss += [loss.cpu().numpy()]
                te_cls_pred.extend(pred.detach().squeeze().cpu().numpy())
                te_pred.extend(pred.detach().squeeze().cpu().numpy())

                if config.loss_type == "cross_entropy":
                    ptc = ptocirc.ptocirc(logits, config.num_class, high=config.max_range, low=config.min_range)
                    te_circ_pred.extend(ptc)

            # all other continuous models
            # TODO get VA model in there
            else:
                loss = data_loss(logits, y) + model_loss(model)

                circ_pred = logits.detach().squeeze()

                cls_circ_pred = preprocess.label_to_class(circ_pred, config.num_map_class,
                                                          config.min_range, config.max_range)

                pred = logits.detach().squeeze()
                acc = torch.mean(torch.eq(cls_circ_pred, y_class).float()) * 100.0

                te_acc += [acc.cpu().numpy()]
                te_loss += [loss.cpu().numpy()]
                te_circ_pred.extend(circ_pred.cpu().numpy())
                te_cls_pred.extend(cls_circ_pred.cpu().numpy())
                te_pred.extend(pred.cpu().numpy())

            te_label.extend(y.cpu().numpy())

        te_cls_label.extend(y_class.cpu().numpy())
        te_circ_label.extend(y.cpu().numpy())

    te_loss = np.mean(te_loss, axis=0)

    print("\nlabel/pred")
    print(np.concatenate(te_label, axis=0)[:10])
    print(te_pred[:10])

    print("\nclass label/pred")
    print(te_cls_label[:10])
    print(te_cls_pred[:10])

    print("\ncirc label/pred")
    print((np.concatenate(te_circ_label, axis=0)*180/np.pi)[:10])
    print(np.array(te_circ_pred[:10])*180/np.pi)

    return te_loss, np.mean(te_acc), te_label, te_pred, te_cls_label, te_cls_pred, te_circ_label, te_circ_pred


def cv_train(config):
    # TODO test changes
    log_dir = config.log_dir
    save_dir = config.save_dir

    for k in range(config.k_fold):
        config.k = k

        config.log_dir = log_dir
        config.save_dir = save_dir

        if config.mode == "train":
            train(config)
            lo, acc, t_y, t_p, cl_y, cl_p, circ_y, circ_p = test(config)

        elif config.mode == "test":
            lo, acc, t_y, t_p, cl_y, cl_p, circ_y, circ_p = test(config)

        else:
            raise ValueError("Unknown run mode \"{}\"".format(config.mode))

        loss = np.array(lo)
        accuracy = np.array(acc)
        test_label = np.array(t_y)
        test_pred = np.array(t_p)
        cls_label = np.array(cl_y)
        cls_pred = np.array(cl_p)
        circ_label = np.array(circ_y)
        circ_pred = np.array(circ_p)

        if k > 0:
            k_loss = np.append(k_loss, loss)
            k_accuracy = np.append(k_accuracy, accuracy)
            k_test_label = np.append(k_test_label, test_label)
            k_test_pred = np.append(k_test_pred, test_pred)
            k_cls_label = np.append(k_cls_label, cls_label)
            k_cls_pred = np.append(k_cls_pred, cls_pred)
            k_circ_label = np.append(k_circ_label, circ_label)
            k_circ_pred = np.append(k_circ_pred, circ_pred)


        else:
            k_loss = loss
            k_accuracy = accuracy
            k_test_label = test_label
            k_test_pred = test_pred
            k_cls_label = cls_label
            k_cls_pred = cls_pred
            k_circ_label = circ_label
            k_circ_pred = circ_pred


    print(k_accuracy)
    # print some report
    evaluate.evaluate_results(config, k_loss, k_accuracy, k_test_label, k_test_pred,
                              k_cls_label, k_cls_pred, k_circ_label, k_circ_pred, mr=False)


def write_to_board(board_stats, writer, iter_idx):
    writer.add_scalar("loss", board_stats["loss"], global_step=iter_idx)
    writer.add_scalar("accuracy", board_stats["acc"], global_step=iter_idx)


def main(config):
    dataset, model_type = path_util.make_pathname(config)

    add_to_path = '/' + config.add_to_path if config.add_to_path != '' else ''

    # results_dir contains h5 file containing accuracy results for all models trained on given dataset
    config.results_dir = config.save_dir + dataset + add_to_path

    config.log_dir = config.log_dir + dataset + add_to_path + model_type
    config.save_dir = config.save_dir + dataset + add_to_path + model_type
    config.model_type = model_type[1:-1]
    print("Main log_dir {}".format(config.log_dir))
    print("Main save_dir {}".format(config.save_dir))
    print("\nTraining on {}".format(config.train))

    if config.test is None:
            config.test = config.train

    print("Testing on {}".format(config.test))

    if config.cv:
        cv_train(config)

    else:
        if config.mode == "train":
            train(config)
            lo, acc, t_y, t_p, cl_y, cl_p, circ_y, circ_p, mad = test(config)

        elif config.mode == "test":
            lo, acc, t_y, t_p, cl_y, cl_p, circ_y, circ_p, mad = test(config)

        else:
            raise ValueError("Unknown run mode \"{}\"".format(config.mode))

        loss = np.array(lo)
        accuracy = np.array(acc)
        test_label = np.array(t_y)
        test_pred = np.array(t_p)
        cls_label = np.array(cl_y)
        cls_pred = np.array(cl_p)
        circ_label = np.array(circ_y)
        circ_pred = np.array(circ_p)

        evaluate.evaluate_results(config, loss, accuracy, test_label, test_pred,
                                  cls_label, cls_pred, circ_label, circ_pred)


if __name__ == "__main__":

    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    print('config.label_type: {}'.format(config.label_type))

    if config.label_type != "class":
        config.num_class = 1

    # # use dataset specific default values when None are specified by args
    # if config.feat == 'raw_audio' or config.label_type == "tricycle":
    #     dsp_param = {'4q': {'sr': 22050, 'n_fft': 1024, 'hop_length': 1024//4,
    #                         'win_length': 1024, 'window': 'hann', 'n_mels': 128, 'duration': 2.0,
    #                         'num_map_class': 4, 'min_range': -np.pi, 'max_range': np.pi},
    #
    #                  'amg1608': {'sr': 22050, 'n_fft': 1024, 'hop_length': 1024 // 4,
    #                              'win_length': 1024, 'window': 'hann', 'n_mels': 128, 'duration': 2.0,
    #                              'num_map_class': 4, 'min_range': -np.pi, 'max_range': np.pi},
    #
    #                  'pmemo': {'sr': 22050, 'n_fft': 1024, 'hop_length': 1024 // 4,
    #                           'win_length': 1024, 'window': 'hann', 'n_mels': 128, 'duration': 2.0,
    #                           'num_map_class': 4, 'min_range': -np.pi, 'max_range': np.pi},
    #
    #                  'deam': {'sr': 22050, 'n_fft': 1024, 'hop_length': 1024 // 4,
    #                              'win_length': 1024, 'window': 'hann', 'n_mels': 128, 'duration': 2.0,
    #                              'num_map_class': 4, 'min_range': -np.pi, 'max_range': np.pi}}
    #
    #     config.sr = dsp_param[config.train]['sr'] if config.sr is None else config.sr
    #     config.n_fft = dsp_param[config.train]['n_fft'] if config.n_fft is None else config.n_fft
    #     config.hop_length = dsp_param[config.train]['hop_length'] if config.hop_length is None else config.hop_length
    #     config.win_length = dsp_param[config.train]['win_length'] if config.win_length is None else config.win_length
    #     config.window = dsp_param[config.train]['window'] if config.window is None else config.window
    #     config.n_mels = dsp_param[config.train]['n_mels'] if config.n_mels is None else config.n_mels
    #     config.duration = dsp_param[config.train]['duration'] if config.duration is None else config.duration
    #
    #     config.min_range = dsp_param[config.train]['min_range']
    #     config.max_range = dsp_param[config.train]['max_range']

    print('config.num_class: {}'.format(config.num_class))

    main(config)
