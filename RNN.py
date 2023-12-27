import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.autograd import Variable


def load_data(file_path, train_ratio = 0.8, seed = 42):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'])
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    train_size = int(len(data) * train_ratio)

    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data
if __name__ == '__main__':
    train_data, test_data = load_data("./SensorMLDataset_small.csv")
    print(test_data)