
def make_pathname(config):
    """ Creates strings to use as paths for logs and save directories

    Creates strings to use as paths for logs and save directories based on dataset, model architecture and parameters

    Parameters
    ----------
    config

    Returns
    -------
    dataset: str
        string from dataset's name and attributes
    model_type: str
        string from model's name and attributes

    """
    if config.cv:
        prefix = '/cv'
    else:
        prefix = ''

    # set dataset
    if '.' in config.train:
        dataset, ext = config.train.split('.')

    else:
        dataset = config.train

    # set optimizer
    amsgrad = "_ams" if (config.amsgrad and config.optim == 'adam') else ""
    mom = ""
    if config.momentum > 0:
        mom = "_nest{}".format(config.momentum) if config.nesterov else "_mome{}".format(config.momentum)

    opt = "_{}{}{}".format(config.optim, amsgrad, mom)

    # set model name
    if config.circ_sync:
        m = "/ensemble{}".format(config.num_cuts)

        if not config.rotate_bias:
            m = m + '_no_rot'

    else:
        if config.num_class == 1:
            m = "/linreg"
        else:
            m = "/class"

    if config.loss_type == "circ_mse":
        if config.num_class != 1:
            raise RuntimeError("Loss \"{}\" is ill-suited for num_class = {}".format(config.loss_type, config.num_class))
        lo = "_circMSE/"
    elif config.loss_type == "mse":
        if config.num_class != 1:
            raise RuntimeError("Loss \"{}\" is ill-suited for num_class = {}".format(config.loss_type, config.num_class))
        lo = "_MSE/"
    elif config.loss_type == "svm":
        if config.num_class == 1:
            raise RuntimeError("Loss \"{}\" is ill-suited for num_class = {}".format(config.loss_type, config.num_class))
        lo = "_SVM_{}cl/".format(config.num_class)
    elif config.loss_type == "cross_entropy":
        if config.num_class == 1:
            raise RuntimeError("Loss \"{}\" is ill-suited for num_class = {}".format(config.loss_type, config.num_class))
        lo = "_cross_ent_{}cl/".format(config.num_class)
    else:
        raise RuntimeError("Loss is not defined or can't be retrieved from config")

    lr = "_lr{}".format(round(config.lr, 6))

    model_type = prefix + m + opt + lr + lo

    return dataset, model_type


if __name__ == "__main__":

    print("This file should be use to append information to logs and save path")
