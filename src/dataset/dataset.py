import itertools
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import IterableDataset, DataLoader


class MultiDisDsprites(IterableDataset):
    """Store dsprites images"""

    def __init__(self,
                 path='/home/akorchemnyi/multi-dis-dsprites/src/dataset/data/dsprite_train.npz',
                 size: int = 10 ** 5):
        dataset_zip = np.load(path)
        # data
        self.imgs = dataset_zip['imgs']
        self.labels = dataset_zip['latents_classes'][:, 1:]

        # feature info
        self.dsprites_size = len(self.imgs)
        self.size = size
        self.lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
        self.features_count = [3, 6, 40, 32, 32]
        self.features_range = [list(range(i)) for i in self.features_count]
        self.multiplier = list(itertools.accumulate(self.features_count[-1:0:-1], operator.mul))[::-1] + [1]
        # print(self.multiplier)

    def get_element_pos(self, labels: np.ndarray) -> int:
        pos = 0
        for mult, label in zip(self.multiplier, labels):
            pos += mult * label
        return pos

    def __iter__(self):
        return self.sample_generator()

    def sample_generator(self):
        for i in range(self.size):
            yield self.generate_sample()

    def get_pair(self):
        """get random pair of objects that differ only in one feature"""
        idx = random.randint(0, self.dsprites_size - 1)

        img = self.imgs[idx]
        label = self.labels[idx]

        feature_type = random.randrange(0, 5)

        exchange_labels = np.zeros_like(label, dtype=bool)
        exchange_labels[feature_type] = True

        other_feature = random.choice(self.features_range[feature_type])
        while other_feature == label[feature_type]:
            other_feature = random.choice(self.features_range[feature_type])
            continue

        other_label = label[:]
        other_label[feature_type] = other_feature

        pair_idx = self.get_element_pos(other_label)
        pair_img = self.imgs[pair_idx]

        # img = torch.from_numpy(img).float().unsqueeze(0)
        # pair_img = torch.from_numpy(pair_img).float().unsqueeze(0)
        # exchange_labels = torch.from_numpy(exchange_labels).unsqueeze(-1)

        # img -> (1, 64, 64)
        img = np.expand_dims(img, 0)
        # pair_img ->(1, 64, 64)
        pair_img = np.expand_dims(pair_img, 0)
        # exchange_labels -> (5, 1)
        exchange_labels = np.expand_dims(exchange_labels, -1)

        return img, pair_img, exchange_labels

    def generate_object(self, scene1, scene2) -> ndarray:
        while True:
            # select random image
            n = random.randint(0, self.dsprites_size - 1)
            obj = np.expand_dims(self.imgs[n], 0)

            # if image intersect scene, try find next
            if np.any(scene1 & obj | scene2 & obj):
                continue
            return obj

    def generate_sample(self):
        # empty scene
        scene1 = np.zeros((1, 64, 64), dtype=int)
        scene2 = np.zeros((1, 64, 64), dtype=int)

        # get random pair of objects that differ only in one feature
        first_obj, pair_obj, exchange_labels = self.get_pair()

        # start creating scenes
        scene1 += first_obj
        scene2 += pair_obj

        # second_obj -> (1, 64, 64)
        second_obj = self.generate_object(scene1, scene2)
        scene1 += second_obj
        scene2 += second_obj

        scene1 = torch.from_numpy(scene1).float()
        scene2 = torch.from_numpy(scene2).float()
        first_obj = torch.from_numpy(first_obj).float()
        second_obj = torch.from_numpy(second_obj).float()
        pair_obj = torch.from_numpy(pair_obj).float()

        return scene1, scene2, first_obj, pair_obj, second_obj, exchange_labels

    def inference_sample(self):
        # empty scene where to add objects
        scene = np.zeros((1, 64, 64), dtype=int)

        # store separate objects (1, 64, 64)
        objs = []

        # Store label info
        # Color: white
        # Shape: square, ellipse, heart
        # Scale: 6 values linearly spaced in [0.5, 1]
        # Orientation: 40 values in [0, 2 pi]
        # Position X: 32 values in [0, 1]
        # Position Y: 32 values in [0, 1]

        # contains info if it empty image for consistence
        masks = []

        # number of objects on scene
        n_objs = 3
        for i in range(n_objs):
            obj = self.generate_object(scene, scene)
            scene += obj.astype(int)
            objs.append(obj)

        # stack elements into torch tensors
        scene = torch.from_numpy(scene).float()
        image1, donor, image2 = objs
        image1 = torch.from_numpy(image1).float()
        donor = torch.from_numpy(donor).float()
        image2 = torch.from_numpy(image2).float()

        return scene, image1, donor, image2


if __name__ == '__main__':
    # dataset
    mdd = MultiDisDsprites(path='/home/yessense/PycharmProjects/multi-dis-dsprites/src/dataset/data/dsprite_train.npz')


    def show_pairs(mdd: MultiDisDsprites, sample_size: int = 5):
        fig, ax = plt.subplots(sample_size, 2)
        for i in range(sample_size):
            img, pair, exchange_labels = mdd.get_pair()
            ax[i, 0].imshow(img.squeeze(0), cmap='gray')
            ax[i, 1].imshow(pair.squeeze(0), cmap='gray')

        plt.show()


    def show_inference_dataset(mdd: MultiDisDsprites, sample_size: int = 5):
        fig, ax = plt.subplots(sample_size, 4)
        for i in range(sample_size):
            scene, image1, donor, image2 = mdd.inference_sample()
            ax[i, 0].imshow(scene.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 1].imshow(image1.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 2].imshow(donor.detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[i, 3].imshow(image2.detach().cpu().numpy().squeeze(0), cmap='gray')

        for i in range(sample_size):
            for j in range(4):
                ax[i, j].set_axis_off()

        plt.show()


    def show_training_dataset(mdd: MultiDisDsprites, batch_size: int = 5):
        batch_size = 5
        loader = DataLoader(mdd, batch_size=batch_size)

        for i, batch in enumerate(loader):
            scenes1, scenes2, fist_objs, pair_objs, second_objs, exchange_labels = batch
            if i % 4000 == 0:

                fig, ax = plt.subplots(batch_size, 5, figsize=(5, 5))
                for i in range(batch_size):
                    for j, column in enumerate(batch[:-1]):
                        ax[i, j].imshow(column[i].detach().cpu().numpy().squeeze(0), cmap='gray')
                        ax[i, j].set_axis_off()

                plt.show()

            assert torch.all(scenes1 == fist_objs + second_objs)
            assert torch.all(scenes2 == pair_objs + second_objs)


    show_inference_dataset(mdd, 5)