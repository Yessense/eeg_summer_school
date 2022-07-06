import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class PhysionetDataset(Dataset):
    def __init__(self, path_to_directory):
        self.labels_dict = {1: "left", 2: "feet", 3: "right"}
        self.sessions_list = [directory for directory in os.listdir(path_to_directory) if os.path.isdir(directory)]
        self.dataset_path = os.path.join()
        # TODO: Shuffle


    def __getitem__(self, index) -> T_co:
        pass

