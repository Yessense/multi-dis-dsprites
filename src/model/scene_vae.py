from argparse import ArgumentParser
from typing import Tuple, List

import pytorch_lightning as pl
import torch.optim
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn

torch.set_printoptions(sci_mode=False)

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from vsa import *


class MultiDisDspritesVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("MnistSceneEncoder")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(1, 64, 64))  # type: ignore
        parser.add_argument("--latent_dim", type=int, default=1024)
        parser.add_argument("--n_features", type=int, default=5)
        parser.add_argument("--hd_objs", type=bool, default=True)
        parser.add_argument("--hd_features", type=bool, default=True)

        return parent_parser

    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 1024,
                 lr: float = 0.001,
                 n_features: int = 5,
                 dropout: bool = False,
                 hd_objs: bool = False,
                 hd_features: bool = False,
                 feature_names: Optional[List] = None,
                 obj_names: Optional[List] = None,
                 **kwargs):
        super().__init__()
        if feature_names is None:
            feature_names = ['shape', 'size', 'rotation', 'posx', 'posy']
        if obj_names is None:
            obj_names = ['obj1', 'obj2']
        self.step_n = 0
        self.encoder = Encoder(latent_dim=latent_dim, image_size=image_size, n_features=n_features)
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size, n_features=n_features)
        self.img_dim = image_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.save_hyperparameters()
        self.n_features = n_features
        self.hd_objs = hd_objs
        self.hd_features = hd_features

        if self.hd_features:
            features_im: ItemMemory = ItemMemory(name="Features", dimension=self.latent_dim,
                                                 init_vectors=feature_names)
            self.feature_placeholders = torch.Tensor(features_im.memory).float().to(self.device)
            self.feature_placeholders = self.feature_placeholders.unsqueeze(0)
            # size = (1, 5, 1024)
            # ready to .expand()

        if self.hd_objs:
            objs_im: ItemMemory = ItemMemory(name="Objects", dimension=self.latent_dim,
                                             init_vectors=obj_names)
            self.obj_placeholders = [torch.Tensor(objs_im.get_vector(name).vector).float().to(self.device) for name in
                                     obj_names]
            # size =  [1024, 1024]
            # ready to .expand()

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Test mode
            return mu

    def encode_features(self, img):
        mu, log_var = self.encoder(img)
        z = self.reparameterize(mu, log_var)
        z = z.view(-1, 5, self.latent_dim)
        if self.hd_features:
            mask = self.feature_placeholders.expand(z.size()).to(self.device)
            z = z * mask
        return mu, log_var, z

    def encode_scene(self, z1, z2):
        batch_size = z1.shape[0]
        masks = [mask.repeat(batch_size, 1).to(self.device) for mask in self.obj_placeholders]

        # choices -> (-1, 1024)
        choices = torch.randint(2, (batch_size,)).bool().unsqueeze(-1).expand(batch_size, self.latent_dim).to(
            self.device)

        m1 = torch.where(choices, masks[0], masks[1])
        m2 = torch.where(choices, masks[1], masks[0])

        z1 *= m1
        z2 *= m2

        scene = z1 + z2
        return scene

    def training_step(self, batch):
        scene1, scene2, fist_obj, pair_obj, second_obj, exchange_label = batch

        mu1, log_var1, z1 = self.encode_features(fist_obj)
        mu2, log_var2, z2 = self.encode_features(pair_obj)
        mu3, log_var3, z3 = self.encode_features(second_obj)

        mu = (mu1 + mu2 + mu3) / 3
        log_var = (log_var1 + log_var2 + log_var3) / 3

        exchange_label = exchange_label.expand(z1.size())
        # [False, False, False, False, True]
        # Состоит из вектора z2, с одной нужной координатой из z1
        # z1 Восстанавливает 1 изображение
        z1 = torch.where(exchange_label, z1, z2)
        # z2 Восстанавливает 2 изображение изображение
        z2 = torch.where(exchange_label, z2, z1)

        # z1 -> first object -> (-1, 1024)
        z1 = torch.sum(z1, dim=1)
        # z2 -> pair object -> (-1, 1024)
        z2 = torch.sum(z2, dim=1)
        # z3 -> second object -> (-1, 1024)
        z3 = torch.sum(z3, dim=1)

        # multiply by object number placeholders
        scene1_latent = self.encode_scene(z1, z3)
        scene2_latent = self.encode_scene(z2, z3)

        r1 = self.decoder(scene1_latent)
        r2 = self.decoder(scene2_latent)

        total, l1, l2, kld = self.loss_f(r1, r2, scene1, scene2, mu, log_var)

        # log training process
        self.log("total", total, prog_bar=True)
        self.log("l1", l1, prog_bar=True)
        self.log("l2", l2, prog_bar=True)
        self.log("kld", kld, prog_bar=True)
        if self.step_n % 499 == 0:
            # self.logger.experiment.add_image('Scene 1', scene1[0], dataformats='CHW', global_step=self.step_n)
            # self.logger.experiment.add_image('Scene 2', scene2[0], dataformats='CHW', global_step=self.step_n)
            # self.logger.experiment.add_image('Recon 1', r1[0], dataformats='CHW',
            #                                  global_step=self.step_n)
            # self.logger.experiment.add_image('Recon 2', r2[0], dataformats='CHW',
            #                                  global_step=self.step_n)
            # self.logger.experiment.add_image('Fist obj', fist_obj[0], dataformats='CHW',
            #                                  global_step=self.step_n)
            # self.logger.experiment.add_image('Pair obj', pair_obj[0], dataformats='CHW',
            #                                  global_step=self.step_n)
            # self.logger.experiment.add_image('Second obj', second_obj[0], dataformats='CHW',
            #                                  global_step=self.step_n)
            self.logger.experiment.log({
                    "reconstruct/examples": [
                        wandb.Image(scene1[0], caption='Scene 1'),
                        wandb.Image(scene2[0], caption='Scene 2'),
                        wandb.Image(r1[0], caption='Recon 1'),
                        wandb.Image(r2[0], caption='Recon 2'),
                        wandb.Image(fist_obj[0], caption='Object 1'),
                        wandb.Image(pair_obj[0], caption='Pair to O1'),
                        wandb.Image(second_obj[0], caption='Object 2')
                    ]
                })
        self.step_n += 1

        return total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_f(self, r1, r2, scene1, scene2, mu, log_var):
        loss = torch.nn.BCELoss(reduction='sum')
        l1 = loss(r1, scene1)
        l2 = loss(r2, scene2)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if self.step_n % 2 == 0:
            total_loss = l1 + l2 + kld
        else:
            total_loss = l2 + l1 + kld
        return total_loss, l1, l2, kld