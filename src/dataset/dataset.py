import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, IterableDataset
import scipy.signal as sn


def roll2d(a, b, dx=1, dy=1):
    """
    rolling 2d window for nd array
    last 2 dimensions
    parameters
    ----------
    a : ndarray
        target array where is needed rolling window
    b : tuple
        window array size-like rolling window
    dx : int
        horizontal step, abscissa, number of columns
    dy : int
        vertical step, ordinate, number of rows
    returns
    -------
    out : ndarray
        returned array has shape 4
        first two dimensions have size depends on last two dimensions target array
    """
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b[-2]) // dy + 1,) + \
            ((a.shape[-1] - b[-1]) // dx + 1,) + \
            b  # sausage-like shape with 2d cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


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
    def __init__(self, path_to_directory: str,
                 sessions_indices: List[int],
                 used_columns: Optional[List] = None,
                 target_column: str = 'state',
                 dt: int = 256,
                 shift: int = 128,
                 size: int = 10 ** 6):
        self.size = size
        if used_columns is None:
            used_columns = ['F3', 'Fz', 'F4',
                            'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
                            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                            'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
                            'P3', 'Pz', 'P4'
                            ]
        self.used_columns = used_columns
        self.target_column = target_column

        self.path_to_directory = path_to_directory
        self.s_rate = 500

        self.sessions_indices = sessions_indices
        self.session_template = "session_{}"

        self.labels_dict = {1: "left", 2: "feet", 3: "right"}
        self.allowed_labels = list(self.labels_dict.keys())
        self.labels_indicies = list(range(len(self.allowed_labels)))

        self.bci_exp_count = 6
        self.bci_exp_arms = [0, 1, 2]
        self.bci_exp_legs = [3, 4, 5]
        self.bci_exp_template = "bci_exp_{}"
        self.bci_exp_data = "data.csv"

        self.b, self.a = sn.butter(2, [2, 40], btype='bandpass', fs=self.s_rate)
        self.b50, self.a50 = sn.butter(2, [48, 50], btype='bandstop', fs=self.s_rate)

        self.shift = shift
        self.dt = dt

        self.data = [[] for _ in self.allowed_labels]
        self.l_b = 100
        self.u_b = 200

    def __iter__(self):
        for i in range(self.size):
            yield self.generate_item()

    def update_data(self) -> None:
        for label_idx, label in enumerate(self.allowed_labels):
            if len(self.data[label_idx]) < self.l_b:
                while len(self.data[label_idx]) < self.u_b:
                    print(len(self.data[label_idx]))
                    path_to_bci_exp = self.get_random_bci_exp(label)
                    print(path_to_bci_exp)
                    self.data[label_idx].extend(self.read_data(path_to_bci_exp, label))

    def read_data(self, path_to_bci_exp, label):

        data = pd.read_csv(path_to_bci_exp)
        data = data[self.used_columns + [self.target_column]]
        data = data.to_numpy()

        class_change = np.convolve(data[:, -1], [1, -1], 'same') != 0
        class_change = np.roll(class_change.astype(np.int32), -1)
        class_change = np.convolve(class_change, [1, 1], 'same')

        conv = np.sum(roll(class_change, np.ones(self.dt), self.shift), axis=1)
        mask = conv < 2

        rolled = roll2d(data[:, :-1], (self.dt, len(self.used_columns)), 1, self.shift).squeeze()
        x = rolled[mask]
        x = x.transpose(0, 2, 1)

        y = data[:data.shape[0] - self.dt + 1: self.shift, -1]
        y = y[mask]

        shuffled_indices = list(range(len(x)))
        random.shuffle(shuffled_indices)
        return [x[i] for i in shuffled_indices if y[i] == label]

    def generate_item(self):
        self.update_data()
        idx = random.choice(self.labels_indicies)
        x = torch.Tensor(self.data[idx].pop()).float()
        y = torch.Tensor([idx]).int()
        return x, y

    def get_random_bci_exp(self, label):
        session_idx = random.choice(self.sessions_indices)
        session_name = self.session_template.format(session_idx)

        if label == 2:
            bci_exp_idx = random.choice(self.bci_exp_legs)
        else:
            bci_exp_idx = random.choice(self.bci_exp_arms)

        bci_exp_name = self.bci_exp_template.format(bci_exp_idx)

        path_to_bci_exp = os.path.join(self.path_to_directory, session_name, bci_exp_name, self.bci_exp_data)
        return path_to_bci_exp


if __name__ == '__main__':
    path_to_directory = "/home/yessense/Downloads/data_physionet"
    dataset = PhysionetDataset(path_to_directory, [1, 2], dt=256, shift=128)
    # for i in range(1):
    #     x, y = dataset.generate_item()
    #
    #     print(x.shape)
    #     print(y.shape)
    #
    #     exit(0)

    dataloader = DataLoader(dataset, batch_size=10)

    batch = next(iter(dataloader))
    print("Done")
