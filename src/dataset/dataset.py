import os
import random
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, IterableDataset
import scipy.signal as sn

def roll(a, b, dx=1):
    """
    Rolling 1d window on array
    Parameters
    ----------
    a : ndarray
    b : ndarray
        rolling 1D window array. Example np.zeros(64)
    dx : step size (horizontal)
    Returns
    -------
    out : ndarray
        target array
    """
    shape = a.shape[:-1] + (int((a.shape[-1] - b.shape[-1]) / dx) + 1,) + b.shape
    strides = a.strides[:-1] + (a.strides[-1] * dx,) + a.strides[-1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class PhysionetDataset(IterableDataset):
    def __init__(self, path_to_directory: str, sessions_indices: List[int]):
        self.path_to_directory = path_to_directory
        self.s_rate = 500

        self.sessions_indices = sessions_indices
        self.session_template = "session_{}"

        self.labels_dict = {1: "left", 2: "feet", 3: "right"}
        self.allowed_labels = list(self.labels_dict.keys())

        self.bci_exp_count = 6
        self.bci_exp_arms = [0, 1, 2]
        self.bci_exp_legs = [3, 4, 5]
        self.bci_exp_template = "bci_exp_{}"
        self.bci_exp_data = "data.csv"

        self.b, self.a = sn.butter(2, [2, 40], btype='bandpass', fs=self.s_rate)
        self.b50, self.a50 = sn.butter(2, [48, 50], btype='bandstop', fs=self.s_rate)

        self.shift = 128
        self.dt = 256

    def generate_item(self):
        label = random.choice(self.allowed_labels)
        path_to_bci_exp = self.get_random_bci_exp(label)
        data = pd.read_csv(path_to_bci_exp)
        data = data.to_numpy()
        class_change = np.convolve(data[:, -1], [1, -1], 'same') != 0
        class_change = np.roll(class_change.astype(np.int32), -1)
        class_change = np.convolve(class_change, [1, 1], 'same')
        conv = np.sum(roll(class_change, np.ones(self.dt), self.shift), axis=1)

        mask = conv < 2
        rolled = roll2d(data[:, :], (self.dt, column_number), 1, shift).squeeze()
        x = rolled[mask]

        y = arr[:len(arr) - dt + 1:shift, 0]
        y = y[mask]

        return class_change

    def get_random_bci_exp(self, label):
        session_idx = random.choice(self.sessions_indices)
        session_name = self.session_template.format(session_idx)

        if label == 3:
            bci_exp_idx = random.choice(self.bci_exp_legs)
        else:
            bci_exp_idx = random.choice(self.bci_exp_arms)

        bci_exp_name = self.bci_exp_template.format(bci_exp_idx)

        path_to_bci_exp = os.path.join(self.path_to_directory, session_name, bci_exp_name, self.bci_exp_data)
        return path_to_bci_exp


if __name__ == '__main__':
    path_to_directory = "/home/yessense/Downloads/data_physionet"
    dataset = PhysionetDataset(path_to_directory, [1, 2])
    for i in range(10):
        item = dataset.generate_item()
        print(item)
