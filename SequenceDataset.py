import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length):
        self.df = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        self.columns_mean = {}
        self.columns_std = {}
        self.columns_min = {}
        self.columns_max = {}
        self.normalize_method = 'min_max'

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

    def normalize_features_and_target(self, imported_dicts=None, method='min_max'):
        self.normalize_method = method
        if method == 'min_max':
            if imported_dicts is None:
                for c in self.df.columns:
                    self.columns_min[c] = self.df[c].min()
                    self.columns_max[c] = self.df[c].max()
            else:
                max_dict, min_dict = imported_dicts
                self.columns_max = max_dict if self.columns_max == {} else self.columns_max
                self.columns_min = min_dict if self.columns_min == {} else self.columns_min
            for c in self.df.columns:
                self.df[c] = (self.df[c] - self.columns_min[c]) / (self.columns_max[c] - self.columns_min[c])
            self.y = torch.tensor(self.df[self.target].values).float()
            self.X = torch.tensor(self.df[self.features].values).float()
        else:
            if imported_dicts is None:
                for c in self.df.columns:
                    self.columns_mean[c] = self.df[c].mean()
                    self.columns_std[c] = self.df[c].std()
            else:
                mean_dict, std_dict = imported_dicts
                self.columns_mean = mean_dict if self.columns_mean == {} else self.columns_mean
                self.columns_std = std_dict if self.columns_std == {} else self.columns_std
            for c in self.df.columns:
                self.df[c] = (self.df[c] - self.columns_mean[c]) / self.columns_std[c]
            self.y = torch.tensor(self.df[self.target].values).float()
            self.X = torch.tensor(self.df[self.features].values).float()

    def invert_normalization(self, vec, column):
        if self.normalize_method == "min_max":
            return vec * (self.columns_max[column] - self.columns_min[column]) + self.columns_min[column]
        else:
            return vec * self.columns_std[column] + self.columns_mean[column]

    def get_normalization_dicts(self):
        if self.normalize_method == 'min_max':
            return self.columns_min, self.columns_max
        else:
            return self.columns_mean, self.columns_std
