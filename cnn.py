import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 两层卷积，两层池化，两层全连接
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        show_feature_maps(x)
        x = self.conv1(x)
        show_feature_maps(x)
        x = F.relu(x)
        show_feature_maps(x)
        x = self.pool1(x)
        show_feature_maps(x)
        x = self.conv2(x)
        show_feature_maps(x)
        x = F.relu(x)
        show_feature_maps(x)
        x = self.pool2(x)
        show_feature_maps(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show_feature_maps(tensor4d):
    feature_maps = tensor4d.detach().cpu().numpy()
    feature_maps = feature_maps.squeeze(0)
    num_channels = feature_maps.shape[0]

    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 2, 5))

    # 确保 `axes` 是一个数组
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(feature_maps[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Channel {i}")

    plt.show()



if __name__ == "__main__":
    net = LeNet()
    print(net)
    img = Image.open("dog_test.png")
    img = img.convert("L")
    img = img.resize((32, 32))
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img)
    x = x.unsqueeze(0)
    # x = torch.randn(1, 1, 32, 32)
    print(x.shape)
    y = net(x)
    print(y.shape)