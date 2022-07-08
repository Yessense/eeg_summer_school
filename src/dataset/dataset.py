import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, IterableDataset
import scipy.signal as sn
import time


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


class DatasetCreator():
    def __init__(self,
                 path_to_dir: str,
                 used_columns: Optional[List] = None,
                 target_column: str = 'state',
                 dt: int = 256,
                 bci_exp_numbers=(0, 1, 2, 3, 4, 5),
                 val_exp_numbers: Optional[List[int]] = None,
                 used_classes=(1, 2, 3)):
        if used_columns is None:
            used_columns = ['F3', 'Fz', 'F4',
                            'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
                            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                            'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
                            'P3', 'Pz', 'P4'
                            ]
        self.path_to_dir = path_to_dir
        self.used_columns = used_columns
        self.target_column = target_column
        self.dt = dt
        self.s_rate = 128
        self.b, self.a = sn.butter(2, [5, 36], btype='bandpass', fs=self.s_rate)
        self.b50, self.a50 = sn.butter(2, [37, 50], btype='bandstop', fs=self.s_rate)

        self.session_template = "session_{}"
        self.bci_exp_template = "bci_exp_{}"
        self.bci_exp_data = "data.csv"
        self.bci_exp_numbers = bci_exp_numbers
        self.val_exp_numbers = val_exp_numbers

        self.bci_exp_filename = "data.csv"
        self.used_columns = used_columns
        self.used_classes = used_classes

    def create_dataset(self, session_numbers: List[int],
                       shift: int = 128,
                       validation: bool = False):

        start_time = time.time()
        print(f"-" * 40)
        print(f"Creating dataset with parameters:")
        print(f"\tsession_numbers: {session_numbers}")
        print(f"\tshift: {shift}")
        print(f"\tdt: {self.dt}")
        print(f"\tvalidation: {validation}")
        if validation:
            if self.val_exp_numbers is not None:
                bci_exp_numbers = self.val_exp_numbers
            else:
                bci_exp_numbers = list(set(self.bci_exp_numbers) - set(self.val_exp_numbers))
        else:
            bci_exp_numbers = self.bci_exp_numbers

        dataset_hash = hash(frozenset(session_numbers)) + hash(shift) + hash(validation) + hash(
            frozenset(bci_exp_numbers))
        dataset_hash = str(dataset_hash)

        path_to_dataset = os.path.join(self.path_to_dir, dataset_hash)
        if os.path.exists(path_to_dataset):
            return torch.load(path_to_dataset)

        x_data = []
        y_data = []
        for session in session_numbers:
            for bci_exp_number in bci_exp_numbers:
                session_name = self.session_template.format(session)
                bci_exp_name = self.bci_exp_template.format(bci_exp_number)
                bci_exp_path = os.path.join(self.path_to_dir, session_name, bci_exp_name, self.bci_exp_data)

                experiment_data = pd.read_csv(bci_exp_path)
                experiment_data = experiment_data[self.used_columns + [self.target_column]]
                experiment_data = experiment_data.to_numpy()

                x = experiment_data[:, :-1]
                x = sn.lfilter(self.b, self.a, x, axis=0)
                x = sn.lfilter(self.b50, self.a50, x, axis=0)

                mean = x.mean(axis=0)[np.newaxis, :]
                std = x.std(axis=0)[np.newaxis, :]
                x -= mean
                x /= std
                x = roll2d(x, (self.dt, len(self.used_columns)), 1, shift).squeeze()
                x = x.transpose(0, 2, 1)

                y = experiment_data[:, -1]
                y = y[:y.shape[0] - self.dt + 1: shift]

                class_change = np.convolve(experiment_data[:, -1], [1, -1], 'same') != 0
                class_change = np.roll(class_change.astype(np.int32), -1)
                class_change = np.convolve(class_change, [1, 1], 'same')
                conv = np.sum(roll(class_change, np.ones(self.dt), shift), axis=1)
                mask = conv < 2

                used_classes_mask = np.zeros_like(y, dtype=bool)
                for used_class in self.used_classes:
                    used_classes_mask |= y == used_class

                x_data.append(x[mask & used_classes_mask])
                y_data.append(y[mask & used_classes_mask] - 1)

        print(f"Dataset is created. Time elapsed: {time.time() - start_time:0.1f} s.")
        print()
        dataset = torch.tensor(np.concatenate(x_data, axis=0)).float(), \
                  torch.tensor(np.concatenate(y_data, axis=0)).long()
        torch.save(dataset, path_to_dataset)

        return dataset


class Physionet(Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor):
        assert data.shape[0] == target.shape[0]
        self.size = data.shape[0]
        self.data = data
        self.target = target

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


if __name__ == '__main__':
    path_to_directory = "../../data_physionet"
    creator = DatasetCreator(path_to_dir=path_to_directory)
    dataset = creator.create_dataset(list(range(1, 5)))
    x, y = dataset
    print(x.shape)
    print(y.shape)

    # dataset = PhysionetDataset(path_to_directory, [1, 2], dt=256, shift=128)
    # for i in range(1):
    #     x, y = dataset.generate_item()
    #
    #     print(x.shape)
    #     print(y.shape)
    #
    #     exit(0)

    # dataloader = DataLoader(dataset, batch_size=10)
    #
    # batch = next(iter(dataloader))
    print("Done")
