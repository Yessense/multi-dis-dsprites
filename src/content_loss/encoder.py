from typing import Tuple

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 6):
        super(Encoder, self).__init__()

        # Layer parameters
        kernel_size = 4
        self.hidden_dim = 1024
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.reshape = (128, kernel_size, kernel_size)

        n_channels = self.image_size[0]

        # Convolutional layers
        cnn_kwargs = dict(kernel_size=kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channels, 16, **cnn_kwargs)
        self.conv2 = nn.Conv2d(16, 32, **cnn_kwargs)
        self.conv3 = nn.Conv2d(32, 64, **cnn_kwargs)
        self.conv4 = nn.Conv2d(64, 128, **cnn_kwargs)
        # self.conv5 = nn.Conv2d(128, 256, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), self.hidden_dim)
        # Fully connected layers for mean and variance
        self.latent_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with activation
        conv1 = self.activation(self.conv1(x))
        # print(f'Conv1 shape: {x.shape}')
        conv2 = self.activation(self.conv2(conv1))
        # print(f'Conv2 shape: {x.shape}')
        conv3 = self.activation(self.conv3(conv2))
        # print(f'Conv3 shape: {x.shape}')
        conv4 = self.activation(self.conv4(conv3))
        # print(f'Conv4 shape: {x.shape}')
        # x = self.activation(self.conv5(x))
        # print(f'Conv5 shape: {x.shape}')

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        # print(f'View shape: {x.shape}')
        x = self.activation(self.lin1(x))
        # print(f'Lin1 shape: {x.shape}')
        # x = self.activation(self.lin2(x))
        # print(f'Lin2 shape: {x.shape}')

        # Fully connected layer for log variance and mean
        # x.shape -> (batch_size, latent_dim * 2)
        # x.shape -> (128, 1024 * 2)
        x = self.latent_layer(x)
        # print(f'Latent shape: {x.shape}')

        return x, conv1, conv2, conv3, conv4
