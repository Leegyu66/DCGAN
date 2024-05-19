import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

from define import *
from model import D, G

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size)

generator = G()
discriminator = D()

adversarial_loss = nn.BCELoss()
adversarial_loss.cuda()

generator.cuda()
discriminator.cuda()

optimizer_G = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

import time


start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.cuda()


        optimizer_G.zero_grad()


        z = torch.randn(128, 100, 1, 1).cuda()


        generated_imgs = generator(z)
        output = discriminator(generated_imgs).reshape(-1)
        g_loss = adversarial_loss(output, torch.ones_like(output))

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        disc_real = discriminator(real_imgs).reshape(-1)
        real_loss = adversarial_loss(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(generated_imgs.detach()).reshape(-1)
        fake_loss = adversarial_loss(disc_fake, torch.zeros_like(disc_fake))
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            save_image(generated_imgs.data[:25], f"{done}.png", nrow=5, normalize=True)

    # 하나의 epoch이 끝날 때마다 로그(log) 출력
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time}]")