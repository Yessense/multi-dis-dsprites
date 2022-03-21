from argparse import ArgumentParser

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.dataset.dataset import MultiDisDsprites
from src.model.scene_vae import MultiDisDspritesVAE

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/yessense/PycharmProjects/multi-dis-dsprites/src/model/checkpoint/epoch=97-step=95745.ckpt')
program_parser.add_argument("--batch_size", type=int, default=5)

# parse input
args = parser.parse_args()


class Experiment:
    def __init__(self, checkpoint_path: str, cuda=True):
        self.device = 'cuda:0'
        self.model = self.load_model_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.dataset = MultiDisDsprites(
            path='/home/yessense/PycharmProjects/multi-dis-dsprites/src/dataset/data/dsprite_train.npz')

    def load_model_from_checkpoint(self, checkpoint_path: str) -> MultiDisDspritesVAE:
        ckpt = torch.load(checkpoint_path)

        hyperparams = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']

        model = MultiDisDspritesVAE(**hyperparams)
        model.load_state_dict(state_dict)
        return model

    def exchange_feature(self, feat1, feat2, n_feature: int):
        exchange_label = torch.ones(1, 5, 1024)
        exchange_label[:, n_feature, :] = 0
        exchange_label = exchange_label.to(self.model.device).bool()

        out = torch.where(exchange_label, feat1, feat2)
        return out

    def get_decoded_scene(self, latent1, latent2):
        scene = self.model.encode_scene(z1=latent1, z2=latent2)

        decoded = self.model.decoder(scene)
        # decoded[decoded >= 0.7] = 1
        # decoded[decoded  = 0

        return decoded

    def plot_list(self, *images):
        assert 0 < len(images) <= 5

        fig, ax = plt.subplots(1, len(images))
        plt.figure(figsize=(20, 8))

        if len(images) == 1:
            ax.imshow(images[1][0].detach().cpu().numpy().squeeze(0), cmap='gray')
            ax.set_axis_off()
        else:
            for i in range(len(images)):
                ax[i].imshow(images[i][0].detach().cpu().numpy().squeeze(0), cmap='gray')
                ax[i].set_axis_off()

        plt.show()

    def exchange_features(self):
        scene, image1, donor, image2 = self.dataset.inference_sample()

        # make (1, 64, 64) -> (1, 1, 64, 64) batch like
        image1 = torch.unsqueeze(image1, dim=0)
        image2 = torch.unsqueeze(image2, dim=0)
        donor = torch.unsqueeze(donor, dim=0)
        scene = torch.unsqueeze(scene, dim=0)

        # go to gpu
        image1 = image1.to(self.model.device)
        image2 = image2.to(self.model.device)
        donor = donor.to(self.model.device)

        # get latent representations
        latent_i1 = self.model.encode_features_latent(image1)
        latent_d = self.model.encode_features_latent(donor)
        latent_i2 = self.model.encode_features_latent(image2)

        self.plot_list(image1, image2, donor, scene)

        fig, ax = plt.subplots(5, 5)
        plt.figure(figsize=(20, 8))

        for i in range(5):
            exchanged_latent = self.exchange_feature(latent_i1, latent_d, n_feature=i)
            exchanged_scene = self.get_decoded_scene(exchanged_latent, latent_i2)

            for j, img in enumerate([exchanged_scene, image1, image2, donor, scene]):
                ax[i, j].imshow(img[0].detach().cpu().numpy().squeeze(0), cmap='gray')
                ax[i, j].set_axis_off()

        plt.show()




if __name__ == '__main__':
    experiment = Experiment(args.checkpoint_path)
    experiment.exchange_features()
    print("Done")
