import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from src import MobileViT
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

model = MobileViT(
    image_size=(28, 28),
    dims=[],
    depths=[2, 4, 3],
    channels=[],
    num_classes=10,
    # expansion=4,
    # kernel_size=3,
    # patch_size=(2, 2),
)

dataset = CIFAR10('../data', train=True, download=True,
                  transform=transforms.ToTensor())
test_data = CIFAR10('../data', train=False, download=True,
                    transform=transforms.ToTensor())
train_data, valid_data = random_split(
    dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))


batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4)
valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

trainer = pl.Trainer(gpus=-1, auto_scale_batch_size=True)

trainer.fit(model, train_loader, valid_loader)
