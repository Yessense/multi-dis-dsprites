from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch.optim

import wandb

torch.set_printoptions(sci_mode=False)

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from vsa import *


class ContentLossVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ConentLossVAE")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(1, 64, 64))  # type: ignore
        parser.add_argument("--latent_dim", type=int, default=5)

        return parent_parser

    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 5,
                 lr: float = 0.001, **kwargs):
        super().__init__()
        self.step_n = 0
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size)
        self.img_dim = image_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.save_hyperparameters()



    def training_step(self, batch):
        scene1, scene2, fist_obj, pair_obj, second_obj, exchange_label = batch

        z, conv1, conv2, conv3, conv4 = self.encoder(scene1)
        recon = self.decoder(z)


        loss = self.loss_f(recon, scene1)

        # log training process
        self.log("loss", loss, prog_bar=True)

        if self.step_n % 499 == 0:
            self.logger.experiment.log({
                    "reconstruct/examples": [
                        wandb.Image(scene1[0], caption='Scene 1'),
                        wandb.Image(recon[0], caption='Recon 1'),
                    ] })
        self.step_n += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, recon, scene1):
        loss = torch.nn.BCELoss(reduction='sum')
        return loss
