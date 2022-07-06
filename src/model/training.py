import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from src.dataset.dataset import PhysionetDataset

sys.path.append("..")

from im_classifier import IMClassifier

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

classifier = IMClassifier(in_channels=27, n_classes=3, lag_backward=256)

wandb_logger = WandbLogger(project='eeg', log_model=True)

path_to_directory = "/home/yessense/Downloads/data_physionet"
dataset = PhysionetDataset(path_to_directory, list(range(1, 70)), dt=256, shift=128)
train_dataloader = DataLoader(dataset, batch_size=2)

profiler = 'simple'
gpus = None # [0]

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=2,
                     profiler=profiler,
                     logger=wandb_logger)

trainer.fit(model=classifier, train_dataloaders=train_dataloader)
