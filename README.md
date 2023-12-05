# pytorch-learn



This is an individual pytorch learning notebook, maybe

---

## 一、Pytorch加载数据

> 深度学习中，数据很多，不能一次性把数据全部放在模型中进行训练。
>
> 所以利用数据加载，进行顺序的打乱、分批、预处理。
>
>  
>
> 数据每行为一个样本，每列为一个特征。

### （1）Dataset类

提供一种方式去获取数据及其label

- 如何获取每一个数据及其label
- 总共有多少数据



数据集组织的两种常见形式：

**① label以文件名的形式表示**

```
- dataset
	- train
        - class_a
            - train_a00000.jpg
            - train_a00001.jpg
            ...
        - class_b
        	- train_b00000.jpg
            - train_b00001.jpg
            ...
```



**② label单独放在在一个txt中**

```
- dataset
	- train
		- class_a_image
			- train_a00000.jpg
			- train_a00001.jpg
			...
		- class_a_label
			- train_a00000.txt
			- train_a00001.txt
			...
		- class_b_image
			- train_b00000.jpg
			- train_b00001.jpg
			...
		- class_b_label
			- train_b00000.txt
			- train_b00001.txt
			...
```







```python
from torch.utils.data import Dataset	# 引入
import os
from PIL import Image

class MyData(Dataset):								# 定义MyData类继承自Dataset类，
    												# 需要重写__getitem__方法，可选重写__len__方法
    
    def __init__(self, root_dir, label_dir): 		# 初始化	此处的self相当于该类的全局变量
    	self.root_dir = root_dir					# 数据集根目录
        self.label_dir = label_dir					# 数据集label
        self.path = os.path.join(self.root_dir, self.label_dir)	# 将根目录和label组合起来成为图片的上级目录
        self.img_path = os.listdir(self.path)		# 把所有图片的路径都列出来
        
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]				
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)				# 需要引入： from PIL import Image
        label = self.label_dir
        return img, label
   	
    def __len__(self):
        return len(self.img_path)
    
    
root_dir = "dataset/train"
a_label_dir = "class_a"
b_label_dir = "class_b"
a_dataset = MyData(root_dir, a_label_dir)	# 这里是两个单独的Dataset类，对应单独的label
b_dataset = MyData(root_dir, b_label_dir)

train_dataset = a_dataset + b_dataset		# train_dataset 是上面两个数据集的集合
```







### Dataloader类

为后面的网络提供不同的数据形式

```
torch.utils.data.DataLoader

dataset: 传入dataset
batch_size: 批大小
shuffle: 是否打乱
num workers: 线程数
```







## MINIST数据集

下载数据集：

```py
minist = MNIST(root='./data', train=True, download=True)
```

举例：

```python
transform = transforms.Compose([
    transforms.ToTensor(),      # 转为tensor，范围0~1
    transforms.Normalize((0.1307,), (0.3081, )) # 归一化
])

train_data = MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_data = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_data, shuffle=False, batch_size=64)
'''
MNIST
	root: 相对路径
	train: 是否是训练集（True->训练集，False->测试集）
	download: 是否下载，第一次选择True，已下载后选择False
	transform: 用作预处理
	（一般训练集打乱，测试集不打乱）
'''
```



下载下来的MINIST数据集是一个元组：

```
tuple(
	<class 'PIL.Image.Image'>,
	<class 'int'>
)
```

显示图像可以：

```python
train_data[0][0].show()
```





### transforms图片预处理

#### 转化为Tensor

**torchvision.transforms.ToTensor**方法：

把 [0, 255] 的PIL.Image文件，shape为(height , width, Channel)，转化为范围0~1.0的Tensor类型，shape为 (channel, height, width)

channel：通道数（黑白为1，彩色为3）

#### 归一化

**torchvision.transforms.Normalize(mean, std)**

mean：均值

std： 方差

>为什么要归一化？
>
>（1）归一化后加快了梯度下降求最优解的速度，从而加快了训练网络的收敛性；
>
>（2）归一化有可能提高精度



```
transform = transforms.Compose([
    transforms.ToTensor(),      # 转为tensor，范围0~1
    transforms.Normalize((0.1307,), (0.3081, )) # 归一化
])
```







## Tensorboard的使用

>pytorch version >= 1.1

 使用方法（Example）：

```python
from torch.utils.tensorboard import SummaryWriter

# create a summary writer with automatically generated folder name.
writer = SummaryWriter()
# folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

# create a summary writer using the specified folder name.
writer = SummaryWriter("my_experiment")
# folder location: my_experiment

# create a summary writer with comment appended.
writer = SummaryWriter(comment="LR_0.1_BATCH_16")
# folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
```



