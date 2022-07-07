import sys
from argparse import ArgumentParser

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.dataset.dataset import PhysionetDataset

sys.path.append("..")

from im_classifier import IMClassifier

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--lower_bracket", type=int, default=5000)
program_parser.add_argument("--upper_bracket", type=int, default=10000)

program_parser.add_argument("--shift", type=int, default=128)
program_parser.add_argument("--dt", type=int, default=256)

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

train, test = train_test_split(list(range(1, 110)), test_size=0.2, random_state=42)

train_dataset = PhysionetDataset(args.dataset_path,
                                 train,
                                 dt=args.lag_backward,
                                 shift=args.shift,
                                 lower_bracket=args.lower_bracket,
                                 upper_bracket=args.upper_bracket)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

test_dataset = PhysionetDataset(args.dataset_path,
                                test,
                                dt=args.lag_backward,
                                shift=args.shift,
                                lower_bracket=args.lower_bracket,
                                upper_bracket=args.upper_bracket)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

monitor = 'val loss'
profiler = 'simple'

if args.gpus == 'None':
    gpus = None
elif args.gpus.isnumeric():
    gpus = [int(args.gpus)]
else:
    raise ValueError()

# checkpoint
save_top_k = 2
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     logger=wandb_logger)

trainer.fit(model=classifier, train_dataloaders=train_dataloader)
