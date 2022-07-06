import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

sys.path.append("..")

from im_classifier import IMClassifier

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')


classifier = IMClassifier()

wandb_logger = WandbLogger(project='eeg', log_model=True)

dataset
