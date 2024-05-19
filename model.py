import torch
from torchvision.utils import save_image
from torch import nn

from define import *

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()

        def block(in_channel, out_channel, kernel_size, strides, padding=1, normalize=True):
            layers = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size, strides, padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, dim_size*4, 4, 1, padding=0),
            *block(dim_size*4, dim_size*8, 4, 2),
            *block(dim_size*8, dim_size*4, 4, 2),
            *block(dim_size*4, 3, 4, 2, normalize=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)

        return img

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        def block(in_channel, out_channel, kernel_size, strides, padding=1, normalize=True):
            layers = [nn.Conv2d(in_channel, out_channel, kernel_size, strides, padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channel, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3, dim_size*4, 4, 2, normalize=False),
            *block(dim_size*4, dim_size*8, 4, 2),
            *block(dim_size*8, dim_size*4, 4, 2),
            nn.Conv2d(dim_size*4, 3, 4, 2, 0),
            nn.Sigmoid()
        )

    def forward(self, img):
        output = self.model(img)
        return output

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 32, 32
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = D()
    assert disc(x).shape == (N, 3, 1, 1), "Discriminator test failed"
    gen = G()
    z = torch.randn((N, noise_dim, 1, 1))
    print(gen(z).shape)
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()