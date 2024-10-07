import numpy as np
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

if __name__ == '__main__':
    dog_img = Image.open('./dog_test.png')
    dog_img = dog_img.resize((256, 256))
    dog_img.show()
    transform = transforms.ToTensor()
    tensor = transform(dog_img)
    print('i1: ', tensor.shape)
    tensor = tensor.unsqueeze(0)
    print('i2: ', tensor.shape)

    conv = nn.Conv2d(3, 3, 3, padding=1)
    # conv = nn.Conv2d(3, 1, 3, padding=1)
    feature_map = conv(tensor)

    print('o1: ', feature_map.shape)

    feature_map = feature_map.squeeze(0)
    # feature_map = feature_map.permute(1, 2, 0)
    print('o2: ', feature_map.shape)
    feature_map = transforms.ToPILImage()(feature_map)
    feature_map.show()