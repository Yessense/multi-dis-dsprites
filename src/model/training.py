import torchvision
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from pytorch_lightning.loggers import WandbLogger

from src.dataset.dataset import MultiDisDsprites
from src.model.scene_vae import MultiDisDspritesVAE
import pytorch_lightning as pl
from argparse import ArgumentParser

wandb_logger = WandbLogger(project='multi-dis-dsprites-vae')

# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--mnist_download_dir", type=str,
                            default='/home/yessense/PycharmProjects/mnist_scene/mnist_download')
program_parser.add_argument("--dataset_size", type=int, default=10 ** 6)
program_parser.add_argument("--batch_size", type=int, default=256)

# add model specific args
parser = MultiDisDspritesVAE.add_model_specific_args(parent_parser=parser)

# add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

iterable_dataset = MultiDisDsprites(size=args.dataset_size)
loader = DataLoader(iterable_dataset, batch_size=args.batch_size, num_workers=1)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)
autoencoder = MultiDisDspritesVAE(**dict_args)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'combined_loss'

# early stop
patience = 5
early_stop_callback = EarlyStopping(monitor=monitor, patience=patience)

# checkpoint
save_top_k = 2
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

callbacks = [
    checkpoint_callback,
    # early_stop_callback,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
max_epochs = 220
gpus = 1

# trainer
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     limit_val_batches=0.0,
                     logger=wandb_logger)
trainer.fit(autoencoder, loader)
