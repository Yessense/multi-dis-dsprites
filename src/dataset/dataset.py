import random
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from torch.utils.data import IterableDataset, DataLoader, Dataset
import itertools
import operator


class MultiDisDsprites(IterableDataset):
    """Store dsprites images"""

    def __init__(self,
                 path='/home/yessense/PycharmProjects/dsprites-disentanglement/src/dataset/data/dsprite_train.npz',
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

        while (other_feature := random.choice(self.features_range[feature_type])) == label[feature_type]: pass
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


if __name__ == '__main__':
    # dataset
    mdd = MultiDisDsprites()

    batch_size = 5
    loader = DataLoader(mdd, batch_size=batch_size)

    for i, batch in enumerate(loader):
        scenes1, scenes2, fist_objs, pair_objs, second_objs, exchange_labels = batch
        if i % 1000 == 0:


            fig, ax = plt.subplots(batch_size, 5, figsize=(5, 5))
            for i in range(batch_size):
                for j, column in enumerate(batch[:-1]):
                    ax[i, j].imshow(column[i].detach().cpu().numpy().squeeze(0), cmap='gray')
                    ax[i, j].set_axis_off()

            plt.show()

        assert torch.all(scenes1 == fist_objs + second_objs)
        assert torch.all(scenes2 == pair_objs + second_objs)

    print("Done")
