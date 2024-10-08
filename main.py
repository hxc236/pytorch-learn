import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),      # 转为tensor，范围0~1
    transforms.Normalize((0.1307,), (0.3081, )) # 归一化
])

max_epochs = 10


if __name__=='__main__':
    # load_Dataset
    train_data = MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

