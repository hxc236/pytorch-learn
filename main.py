from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):  # 继承自Dataset类

    def __init__(self, root_dir):
        self.root_dir = root_dir
        for i in range(1, 9):
            self.path[i] = os.path.join(self.root_dir, str(i))
            self.img_path[i] = os.listdir(self.path[i])


    def __getitem__(self, idx):

        pass