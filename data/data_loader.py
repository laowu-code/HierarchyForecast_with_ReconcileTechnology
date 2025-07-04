import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class data_detime(Dataset):
    def __init__(self, data, lookback_length, lookforward_length, multi_steps=True):
        self.seq_len = lookback_length
        self.pred_len = lookforward_length
        self.multi_steps = multi_steps
        self.data_y = data
        self.data=data
        print(self.data.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        x = self.data[s_begin:s_end]
        # x_else = self.data_else[s_begin:s_end]
        if self.multi_steps:
            y = self.data_y[s_end:s_end + self.pred_len, 0]
        else:
            y = self.data_y[s_end + self.pred_len - 1:s_end + self.pred_len, 0]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

def split_data(data, train, test):
    # data = data.loc[~(data['gen'] == 0)]
    # for column in list(data.columns[data.isnull().sum() > 0]):
    #     data[column].interpolate(method='linear', limit_direction='forward')
    # timestamp = data[['date']]
    # timestamp['date'] = pd.to_datetime(timestamp.date)
    # cols = list(data.columns)
    # cols.remove('date')
    # data = data[cols].values
    length = len(data)
    num_train = int(length * train) if train < 1 else train
    num_test = int(length * test) if test < 1 else test
    num_valid = length - num_test - num_train
    # timestamp_train = timestamp[0:num_train]
    # timestamp_valid = timestamp[num_train:num_train + num_valid]
    # timestamp_test = timestamp[num_train + num_valid:]
    # scalar = StandardScaler()
    # scalar_y = StandardScaler()
    scalar_y = MinMaxScaler()
    y = data[0:num_train].reshape(-1, 1)
    scalar_y.fit(y)
    data = scalar_y.transform(data.reshape(-1, 1))
    # data_re=scalar.inverse_transform(data)
    data_train = data[0:num_train]
    data_valid = data[num_train:num_train + num_valid]
    data_test = data[num_train + num_valid:]
    return data_train, data_valid, data_test, scalar_y








