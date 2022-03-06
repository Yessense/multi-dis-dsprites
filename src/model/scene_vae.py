from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch.optim

import wandb

from src.content_loss.scene_vae import ContentLossVAE

torch.set_printoptions(sci_mode=False)

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from vsa import *


class MultiDisDspritesVAE(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("MultiDisDspritesVAE")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--image_size", type=Tuple[int, int, int], default=(1, 64, 64))  # type: ignore
        parser.add_argument("--latent_dim", type=int, default=1024)
        parser.add_argument("--n_features", type=int, default=5)
        parser.add_argument("--hd_objs", type=bool, default=True)
        parser.add_argument("--hd_features", type=bool, default=True)
        parser.add_argument("--content_loss_path", type=str,
                            default='/home/akorchemnyi/multi-dis-dsprites/src/content_loss/content_loss_model.ckpt')

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
                 content_loss_path=None,
                 **kwargs):
        super().__init__()

        if feature_names is None:
            feature_names = ['shape', 'size', 'rotation', 'posx', 'posy']
        if obj_names is None:
            obj_names = ['obj1', 'obj2']
        # if content_loss_path is not None:
        #     self.cl_model = self.load_cl_model(content_loss_path)

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

    def content_loss(self, scene, reconstructed):
        _, conv1_s, conv2_s, conv3_s, conv4_s = self.cl_model.encoder(scene)
        _, conv1_r, conv2_r, conv3_r, conv4_r = self.cl_model.encoder(reconstructed)
        loss = torch.nn.MSELoss()
        loss_1 = loss(conv1_r, conv1_s)
        loss_2 = loss(conv2_r, conv2_s)
        loss_3 = loss(conv3_r, conv3_s)
        loss_4 = loss(conv4_r, conv4_s)

        return loss_1, loss_2, loss_3, loss_4

    def training_step(self, batch):
        scene1, scene2, fist_obj, pair_obj, second_obj, exchange_label = batch
        batch_size = scene1.shape[0]

        mu1, log_var1, feat_1 = self.encode_features(fist_obj)
        mu2, log_var2, feat_2 = self.encode_features(pair_obj)
        mu3, log_var3, z3 = self.encode_features(second_obj)

        mu = (mu1 + mu2 + mu3) / 3
        log_var = (log_var1 + log_var2 + log_var3) / 3

        # exchange_label = exchange_label.expand(z1.size())

        # # ----------------------------------------------------------------------
        # # Exchange feature by cosine similarity
        # # ----------------------------------------------------------------------
        #
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cos(feat_1, feat_2)
        lowest = torch.argmin(torch.abs(similarity), keepdim=False, dim=1)

        labels = torch.zeros(batch_size, self.n_features).bool().to(self.device)
        labels[torch.arange(batch_size), lowest] = True

        exchange_labels = labels.unsqueeze(-1).expand(feat_1.size())

        # exchange_label = torch.zeros(batch_size, self.n_features).bool()
        # exchange_label[torch.arange(batch_size), lowest] = True
        # exchange_label = exchange_label.unsqueeze(-1).expand(z1.size())

        # [False, False, False, False, True]
        # Состоит из векторов False и одного вектора True для самого отличающегося признака.
        # Состоит из вектора z2, с одной нужной координатой из z1
        # z1 Восстанавливает 1 изображение
        z1 = torch.where(exchange_label, feat_1, feat_2)
        # z2 Восстанавливает 2 изображение изображение
        z2 = torch.where(exchange_label, feat_2, feat_1)

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

        total, l1, l2, kld, cos_loss = self.loss_f(r1, r2, scene1, scene2, mu, log_var, feat_1, feat_2, labels)
        iou1 = self.iou_pytorch(r1, scene1)
        iou2 = self.iou_pytorch(r2, scene2)
        iou = (iou1 + iou2) / 2

        # log training process
        self.log("Sum of losses", total, prog_bar=True)
        self.log("BCE reconstruct 1, img 1", l1, prog_bar=False)
        self.log("BCE reconstruct 2, img 2", l2, prog_bar=False)
        self.log("KL divergence", kld, prog_bar=True)
        self.log("IOU mean ", iou, prog_bar=True)
        self.log("IOU reconstruct 1, img 1", iou1, prog_bar=False)
        self.log("IOU reconstruct 2, img 2", iou2, prog_bar=False)
        self.log("Cosine loss", cos_loss, prog_bar=True)

        # loss_1, loss_2, loss_3, loss_4 = self.content_loss(r1, scene1)
        # loss_12, loss_22, loss_32, loss_42 = self.content_loss(r2, scene2)
        # self.log("Content loss recon 1 conv 1", loss_1)
        # self.log("Content loss recon 1 conv 2", loss_2)
        # self.log("Content loss recon 1 conv 3", loss_3)
        # self.log("Content loss recon 1 conv 4", loss_4)
        #
        # self.log("Content loss recon 2 conv 1", loss_12)
        # self.log("Content loss recon 2 conv 2", loss_22)
        # self.log("Content loss recon 2 conv 3", loss_32)
        # self.log("Content loss recon 2 conv 4", loss_42)

        if self.step_n % 499 == 0:
            self.logger.experiment.log({
                "reconstruct/examples": [
                    wandb.Image(scene1[0], caption='Scene 1'),
                    wandb.Image(scene2[0], caption='Scene 2'),
                    wandb.Image(r1[0], caption='Recon 1'),
                    wandb.Image(r2[0], caption='Recon 2'),
                    wandb.Image(fist_obj[0], caption='Image 1'),
                    wandb.Image(pair_obj[0], caption='Pair to Image 1'),
                    wandb.Image(second_obj[0], caption='Image 2')
                ]})
        self.step_n += 1

        return total

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def iou_pytorch(self, outputs: torch.Tensor, labels: torch.Tensor):
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs > 0.5
        outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1).byte()
        SMOOTH = 1e-8
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded.mean()

    def load_cl_model(self, checkpoint_path: str) -> ContentLossVAE:
        """ Load content loss model"""

        ckpt = torch.load(checkpoint_path)

        hyperparams = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']

        model = ContentLossVAE(**hyperparams)
        model.load_state_dict(state_dict)
        return model

    def loss_f(self, r1, r2, scene1, scene2, mu, log_var, feat_1, feat_2, exchange_labels):
        loss = torch.nn.BCELoss(reduction='sum')
        l1 = loss(r1, scene1)
        l2 = loss(r2, scene2)

        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Cos Loss
        cos_loss = torch.tensor([0.]).to(self.device)
        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(reduction='sum')
        target = torch.logical_not(exchange_labels).float() * 2. - 1.
        for i in range(self.n_features):
            curr_feat1 = feat_1[:, i]
            curr_feat2 = feat_2[:, i]
            curr_target = target[:, i]
            curr_cos_loss = cosine_embedding_loss(curr_feat1, curr_feat2, curr_target)
            cos_loss += curr_cos_loss

        total_loss = l1 + l2 + kld * 0.01 + cos_loss
        return total_loss, l1, l2, kld, cos_loss
