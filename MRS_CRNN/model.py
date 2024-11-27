# model.py

import torch
import torch.nn as nn
import math

class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class):
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width)

        self.rec1 = nn.GRU(output_channel * output_height * output_width, 10, bidirectional=True, batch_first=True)
        self.rec2 = nn.GRU(20, num_class, bidirectional=False, batch_first=True)
        self.drop_final = nn.Dropout(0.3)
        self.dense = nn.Linear(num_class, num_class)  # Adjusted to output num_class

    def _cnn_backbone(self, img_channel, img_height, img_width):
        channels = [img_channel, 64, 128, 128, 128, 128]
        kernel_size = (3, 3)
        pool_sizes = [(2, 2), (4, 2), (4, 2), (4, 2), (1, 2)]

        cnn = nn.Sequential()

        def conv_elu(i):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_size, padding=1)
            )
            cnn.add_module(f'elu{i}', nn.ELU())
            cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))
            cnn.add_module(f'pooling{i}', nn.MaxPool2d(kernel_size=pool_sizes[i]))
            cnn.add_module(f'dropout{i}', nn.Dropout(0.1))

        cnn.add_module(f'batchnorm_initial', nn.BatchNorm2d(img_channel))
        for i in range(len(pool_sizes)):
            conv_elu(i)

        def compute_dims(h_w, kernel_dim):
            h, w = h_w
            h_new = math.floor((h - (kernel_dim[0] - 1) - 1) / kernel_dim[0] + 1)
            w_new = math.floor((w - (kernel_dim[1] - 1) - 1) / kernel_dim[1] + 1)
            return h_new, w_new

        output_channel = channels[-1]
        output_height, output_width = img_height, img_width
        for pool_size in pool_sizes:
            output_height, output_width = compute_dims((output_height, output_width), pool_size)

        return cnn, (output_channel, output_height, output_width)

    def forward(self, images, return_features=False):
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, -1).unsqueeze(1)
        conv, _ = self.rec1(conv)
        conv, _ = self.rec2(conv)
        conv = self.drop_final(conv)  # Dropout layer output (latent features)
        
        if return_features:
            return conv[:, -1, :]  # Return latent features from drop_final for recommendation
        else:
            conv = self.dense(conv[:, -1, :])  # Final classification layer for training/testing
            return conv

