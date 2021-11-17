import argparse
import numpy as np

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for argparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test", "tuning"],
                      help="Run mode")

main_arg.add_argument("--add_to_path", type=str,
                       default='',
                       help="adds <str> to logs and save path. Keeping track of small commit effect on"
                            " experiments' results")

# ----------------------------------------
# Arguments for the dataset
data_arg = add_argument_group("Dataset")

data_arg.add_argument("--train", type=str,
                      default="CircSynth",
                      choices=["circMood.csv", "circMood.h5"],
                      help="Path to training dataset file. "
                           "When using circMood, include extension e.g. circMood.h5")

data_arg.add_argument("--test", type=str,
                      default=None,
                      help="Path to testing dataset file. "
                            "Training dataset is used if not specified")

data_arg.add_argument("--feat", type=str,
                      default="vector",
                      choices=["mel", "vector"],
                      help="Type of features to be used")

data_arg.add_argument("--label_type", type=str,
                      default="class",
                      choices=["class", "circ", "va", "arou", "val"],
                      help="Type of ground truth label")

data_arg.add_argument("--num_class", type=int,
                      default=5,
                      help="Number of classes in the dataset. "
                           "Regression is considered as 1 class "
                           "while classification has multiple classes")

data_arg.add_argument("--num_map_class", type=int,
                      default=5,
                      help="Number of classes to map to in regression models. "
                           "Regression predictions are mapped to num_map_class")

data_arg.add_argument("--num_tag", type=int,
                      default=40,
                      help="Number of tags in the dataset. "
                           "Can also be seen as classification where the classes "
                           "are small intervals of the prediction's circular (linear) domain")

data_arg.add_argument("--min_range", type=float,
                      default=-np.pi,
                      help="Lower bound of the range of continuous data labels")

data_arg.add_argument("--max_range", type=float,
                      default=np.pi,
                      help="Upper bound of the range of continuous data labels")

data_arg.add_argument("--standardize", type=str2bool,
                      default=False,
                      help="Whether to standardize with mean/std or not")

data_arg.add_argument("--one_batch", type=str2bool,
                      default=False,
                      help="Check performance on one batch (debugging)")


# ----------------------------------------
# Arguments for training

train_arg = add_argument_group("Training")

train_arg.add_argument("--lr", type=float,
                       default=3e-04,
                       help="Learning rate")

train_arg.add_argument("--bs", type=int,
                       default=16,
                       help="Batch size")

train_arg.add_argument("--num_e", type=int,
                       default=1,
                       help="Number of epochs, determined by int((config.num_it/bpere) and "
                            "bpere is the ceiling of (size of training data / batch size)")

train_arg.add_argument("--val_intv", type=int,
                       default=100,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=500,
                       help="Report interval")

train_arg.add_argument("--optim", type=str,
                       default="adam",
                       choices=["adam", "sgd"],
                       help="Optimizer algorithm")

train_arg.add_argument("--amsgrad", type=str2bool,
                       default=False,
                       help="amsgrad flag for adam")

train_arg.add_argument("--nesterov", type=str2bool,
                       default=False,
                       help="nesterov flag for sgd")

train_arg.add_argument("--momentum", type=float,
                       default=0.0,
                       help="momentum value for sgd")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs/",
                       help="Log directory")

train_arg.add_argument("--save_dir", type=str,
                       default="./save/",
                       help="Model directory")

train_arg.add_argument("--results_dir", type=str,
                       default="./save/",
                       help="results.h5 directory for all models on specific dataset")

train_arg.add_argument("--resume", type=str2bool,
                       default=False,
                       help="Whether to resume training from existing checkpoint")

train_arg.add_argument("--cv", type=str2bool,
                       default=False,
                       help="Whether to use cross validation")

train_arg.add_argument("--k_fold", type=int,
                       default=10,
                       help="Number of folds in cross validation")

train_arg.add_argument("--k", type=int,
                       default=0,
                       help="Specify fold, if wanting to use a specific fold")

train_arg.add_argument("--precision", type=str2bool,
                       default=False,
                       help="Make tag prediction from probability")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--loss_type", type=str,
                       default="circ_mse",
                       choices=["cross_entropy", "svm", "mse", "circ_mse"],
                       help="Type of data loss to be used")

model_arg.add_argument("--l2_reg", type=float,
                       default=0,
                       help="L2 Regularization strength")

model_arg.add_argument("--circ_sync", type=str2bool,
                       default=False,
                       help="Tune all cuts simultaneously in parallel on individual loss - Ensemble model")

model_arg.add_argument("--rotate_bias", type=str2bool,
                       default=True,
                       help="Rotate biases of ensemble model")

model_arg.add_argument("--num_cuts", type=int,
                       default=5,
                       help="Number of linear models to approximate circular reg")

model_arg.add_argument("--cut", type=int,
                       default=-1,
                       help="The cut used by model in approximation")

model_arg.add_argument("--model_type", type=str,
                       default="",
                       help="Holds model_type info. Place holder: filled by code - ignore")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
