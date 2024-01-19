import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.autograd import Variable


def load_data(file_path, train_ratio = 0.8, seed = 42, df):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df['Timestamp'] = df['Timestamp'].dt.date
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(df))
    train_size = int(len(df) * train_ratio)

    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_data = df.iloc[train_indices]
    test_data = df.iloc[test_indices]

    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = load_data("./SensorMLDataset_small.csv")
    print(test_data)