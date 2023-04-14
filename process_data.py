import pandas as pd
import numpy as np


def get_data(filename):
    data_frame = pd.read_excel(filename, sheet_name="Sheet1")
    labels = np.zeros([data_frame.shape[0], 2], dtype=np.float32)
    train_data = np.zeros([data_frame.shape[0], data_frame.shape[1] - 1], dtype=np.float32)
    # print(train_data.shape)
    index = 0
    for p in data_frame.iterrows():
        if p[1].values[0] > 0.9:
            labels[index][1] = 1.0
        else:
            labels[index][0] = 1.0
        for i in range(train_data.shape[1]):
            train_data[index][i] = p[1].values[i + 1]
        index = index + 1
    print(train_data)
    return train_data, labels


def log_plus_1(data):
    ret = np.log10(data + 1)
    return ret


def clr(mat):
    return 0


if __name__ == '__main__':
    train_data, labels = get_data('./data/fat_data.xlsx')
    print(log_plus_1(train_data))
