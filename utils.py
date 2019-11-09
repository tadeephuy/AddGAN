import random
import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
def VggGenerator():
    d_1, d_2, d_3 = 64, 128, 256
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=d_1, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ## 64x16x16
        nn.Conv2d(in_channels=d_1, out_channels=d_2, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ## 128x8x8
        nn.Conv2d(in_channels=d_2, out_channels=d_3, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_3),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_3, out_channels=d_3, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_3),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_3, out_channels=d_3, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_3),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        

        #### 256x4x4
        

        nn.ConvTranspose2d(in_channels=d_3, out_channels=d_2, kernel_size=2, stride=2, output_padding=1),
        ## 128x9x9
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_2),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),

        nn.ConvTranspose2d(in_channels=d_2, out_channels=d_1, kernel_size=2, stride=2, output_padding=1),
        ## 64x19x19
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=1),
        nn.BatchNorm2d(d_1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=0),
        nn.BatchNorm2d(d_1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),

        nn.ConvTranspose2d(in_channels=d_1, out_channels=3, kernel_size=2, stride=2),
        ## 3x34x34
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3),
        ## 3x32x32
        nn.Tanh()
    )


class VggDiscriminator(nn.Module):

    def __init__(self):
        super(VggDiscriminator, self).__init__()
        d_1, d_2, d_3, d_4 = 64, 128, 256, 512
        self.features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_1, out_channels=d_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## 64x16x16
            nn.Conv2d(in_channels=d_1, out_channels=d_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_2, out_channels=d_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## 128x8x8
            nn.Conv2d(in_channels=d_2, out_channels=d_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_3, out_channels=d_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_3, out_channels=d_3, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## 256x4x4
            nn.Conv2d(in_channels=d_3, out_channels=d_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_4, out_channels=d_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=d_4, out_channels=d_4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ## 512x2x2
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
            nn.ReLU(True),
            nn.Linear(10, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MnistDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        # ignore idx
        input_class = random.randint(0, 9)
        output_class = (input_class + 1) % 10

        input_imgs = glob(os.path.join(self.data_dir, str(input_class)) + "/*")
        output_imgs = glob(os.path.join(self.data_dir, str(output_class)) + "/*")

        input_img = cv2.imread(random.choice(input_imgs), 0)
        output_img = cv2.imread(random.choice(output_imgs), 0)

        input_img = self.preprocess(self.transform(input_img).float())
        output_img = self.preprocess(self.transform(output_img).float())

        return input_img, output_img, input_class, output_class
    
    def preprocess(self, x):
        x = F.pad(x, pad=(2, 2, 2, 2))
        # x = np.expand_dims(x, axis=0)
        return x
    
def format_Gx(x):
    x = x.clone().detach()
    x = (x + 1)/2.0
    return x

def create_figure(in_img, out_img, fake):
    # concat images
    batch_size = in_img.shape[0]
    grid_in = np.transpose(torchvision.utils.make_grid(in_img, batch_size, pad_value=1), (1, 2, 0))
    grid_out = np.transpose(torchvision.utils.make_grid(out_img, batch_size, pad_value=1), (1, 2, 0))
    grid_fake = np.transpose(torchvision.utils.make_grid(fake, batch_size, pad_value=1), (1, 2, 0))
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 5), gridspec_kw={'hspace': 0.6 })
    fig.suptitle('Images')
    ax1.imshow(grid_in)
    ax1.set_title("Real Original")
    ax2.imshow(grid_out)
    ax2.set_title("Real Original + 1")
    ax3.imshow(grid_fake)
    ax3.set_title("Fake Original + 1")
    return fig