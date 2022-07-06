import sys
from argparse import ArgumentParser

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from src.dataset.dataset import PhysionetDataset

sys.path.append("..")

from im_classifier import IMClassifier

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--in_channels", type=int, default=27)
program_parser.add_argument("--n_classes", type=int, default=3)
program_parser.add_argument("--lag_backward", type=int, default=256)
program_parser.add_argument("--shift", type=int, default=128)
program_parser.add_argument("--lower_bracket", type=int, default=5000)
program_parser.add_argument("--upper_bracket", type=int, default=10000)

program_parser.add_argument("--dataset_path", type=str, default="../data_physionet/")
program_parser.add_argument("--batch_size", type=int, default=512)

program_parser.add_argument("--gpus", type=str, default='0')

parser = IMClassifier.add_model_specific_args(parent_parser=parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

classifier = IMClassifier(in_channels=args.in_channels,
                          n_classes=args.n_classes,
                          lag_backward=args.lag_backward)

wandb_logger = WandbLogger(project='eeg', log_model=True)

dataset = PhysionetDataset(args.dataset_path,
                           list(range(1, 70)),
                           dt=args.lag_backward,
                           shift=args.shift,
                           lower_bracket=args.lower_bracket,
                           upper_bracket=args.upper_bracket)
train_dataloader = DataLoader(dataset, batch_size=args.batch_size)

profiler = 'simple'

if args.gpus == 'None':
    gpus = None
elif args.gpus.isnumeric():
    gpus = [int(args.gpus)]
else:
    raise ValueError()

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     logger=wandb_logger)

trainer.fit(model=classifier, train_dataloaders=train_dataloader)
