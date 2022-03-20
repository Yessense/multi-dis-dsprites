from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.dataset.dataset import MultiDisDsprites
from src.model.scene_vae import MultiDisDspritesVAE

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')
program_parser.add_argument("--checkpoint_path", type=str,
                            default='/home/yessense/PycharmProjects/multi-dis-dsprites/src/model/checkpoint/epoch=77-step=76205.ckpt')
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

    def forward_images(self):
        scene, image1, donor, image2 = self.dataset.inference_sample()

        image1 = torch.unsqueeze(image1, dim=0)
        image2 = torch.unsqueeze(image2, dim=0)
        donor = torch.unsqueeze(donor, dim=0)
        scene = torch.unsqueeze(scene, dim=0)

        image1 = image1.to(self.model.device)
        image2 = image2.to(self.model.device)
        donor = donor.to(self.model.device)

        latent_i1 = self.model.encode_features_latent(image1)
        latent_i2 = self.model.encode_features_latent(image2)
        latent_d = self.model.encode_features_latent(donor)

        scene1 = self.model.encode_scene(z1=latent_i1, z2=latent_i2)
        scene2 = self.model.encode_scene(z1=latent_i1, z2=latent_d)
        scene3 = self.model.encode_scene(z1=latent_i2, z2=latent_d)

        decoded1 = self.model.decoder(scene1)
        decoded2 = self.model.decoder(scene2)
        decoded3 = self.model.decoder(scene3)

        fig, ax = plt.subplots(1, 4)
        plt.figure(figsize=(20, 8))

        ax[0].imshow(image1[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[0].set_axis_off()
        ax[1].imshow(image2[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[1].set_axis_off()
        ax[2].imshow(donor[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[2].set_axis_off()
        ax[3].imshow(scene[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[3].set_axis_off()

        plt.show()

        fig, ax = plt.subplots(1, 3)
        plt.figure(figsize=(20, 8))

        ax[0].imshow(decoded1[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[0].set_axis_off()
        ax[1].imshow(decoded2[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[1].set_axis_off()
        ax[2].imshow(decoded3[0].detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[2].set_axis_off()

        plt.show()


if __name__ == '__main__':
    experiment = Experiment(args.checkpoint_path)
    experiment.forward_images()
    print("Done")
