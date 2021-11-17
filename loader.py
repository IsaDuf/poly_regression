import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch, torchaudio
import torch.utils.data as data
import numpy as np
import h5py
# import librosa
# import scipy.stats as st

from utils import preprocess
# rvon_mises

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') if CUDA else torch.device('cpu')


class CircMood(data.Dataset):

    def __init__(self, config, data_type, data_file):

        self.dfile = data_file
        self.feat = config.feat
        self.label_type = config.label_type
        self.split = data_type
        self.k_fold = config.k_fold if config.cv else 1
        self.k = config.k
        self.standardize = config.standardize
        self.cut = config.cut
        self.circ_sync = config.circ_sync

        # If dataset is in CSV file, need to split in k folds and create the h5 file
        # that is used in __get_item__
        if '.csv' in self.dfile:

            self.csv_to_h5(data_file)

            # change data file to newly created h5 file
            self.dfile = "circMood.h5"

        # load specified fold from h5
        with h5py.File("data/circMood.h5", "r") as f:
            self.tids = f["folds"][str(self.k)][self.split][:]
            self.data_mean = f["folds"][str(self.k)]["data_mean"][:]
            self.data_std = f["folds"][str(self.k)]["data_std"][:]
            self.data_shape = np.float64(f["0"]["X"][:]).shape

    def csv_to_h5(self, data_file: str) -> None:
        """ Create h5 file from original csv file

        Create the h5 file that is used in __getitem__.
        The circMood dataset is quite small and requires k_fold training (paper experiments used 10 folds: default).
        Split dataset in k stratified folds and save everything to h5 file.

        Parameters
        ----------
        data_file : str
        csv file (circMood.csv) to be split and save as h5. This method is specific to the csv structure of circMood.csv

        Returns
        -------
        None
        """
        df = pd.read_csv('data/' + data_file)
        X_feat, y_class, y_circ, y_arou, y_val = pd.DataFrame(df.loc[:, df.columns.values[1]:'Beat']), \
                                                 pd.DataFrame(df['output']), pd.DataFrame(df['outputLog']), \
                                                 pd.DataFrame(df['Arousal']), pd.DataFrame(df['Valence'])

        # create stratified fold (seed fixed for reproducibility)
        skf = StratifiedKFold(n_splits=self.k_fold, random_state=294, shuffle=True)
        train_index_list, train_index, val_index, test_index = [], [], [], []
        for train_index_i, test_index_i in skf.split(X_feat, y_class):
            train_index_list.append(train_index_i)
            test_index.append(test_index_i)
        # create validation set from test set on different folds
        # e.g. in fold 0, test set from fold 1 is used as validation
        #      in fold 1, test set from fold 2 is used as validation
        for i in range(1, (self.k_fold + 1)):
            test_ind = i % self.k_fold
            val_index.append(test_index[test_ind])
            train_index.append(np.setdiff1d(train_index_list[i - 1], val_index[i - 1]))
        # create h5 file
        # in a n_xm dataset, the n <tids> groups contain 5 datasets: 'X', 'y_class', y_circ, y_arou, y_val
        with h5py.File("data/circMood.h5", "w") as f:
            for i in range(len(X_feat)):
                track = f.create_group(str(i))
                track.create_dataset("X", data=X_feat.loc[i].values)
                track.create_dataset("y_class", data=y_class.loc[i].values)
                track.create_dataset("y_circ", data=y_circ.loc[i].values)
                track.create_dataset("y_arou", data=y_arou.loc[i].values)
                track.create_dataset("y_val", data=y_val.loc[i].values)

            # the k <folds> groups contain the "train_index", "test_index", "val_index" datasets of each fold
            # and their corresponding "data_mean" and "data_std"
            folds = f.create_group("folds")
            for i in range(self.k_fold):
                fold = folds.create_group(str(i))
                fold.create_dataset("train", data=train_index[i])
                fold.create_dataset("test", data=test_index[i])
                fold.create_dataset("val", data=val_index[i])

                # get mean and std of training data for each fold while we're here:
                all_fold = X_feat.values
                this_fold = np.delete(all_fold, np.append(test_index[i], val_index[i]), 0)
                data_mean = np.mean(this_fold, axis=0).reshape((97,))
                data_std = np.std(this_fold - data_mean, axis=0).reshape(97, )
                fold.create_dataset("data_mean", data=data_mean)
                fold.create_dataset("data_std", data=data_std)

    def __getitem__(self, index):

        stand = self.standardize
        dataf = self.dfile

        with h5py.File("data/"+dataf, "r") as f:
            id = self.tids[index]

            # get features
            if stand:
                X = np.float64(f[str(id)]["X"][:])
                X = preprocess.standardize(X, self.data_mean, self.data_std).astype(np.float32)

            else:
                X = np.float32(f[str(id)]["X"][:])

            # get labels
            if self.label_type == "class":
                y = int(f[str(id)]["y_"+self.label_type][:])-1
                y_class = y

            else:
                y = np.float32(f[str(id)]["y_"+self.label_type][:])
                y = preprocess.scale_to_range(y, -np.pi, np.pi, 1.0, 6.0)
                y_class = int(f[str(id)]["y_class"][:])-1

        return X, y, y_class

    def __len__(self):
        return len(self.tids)
