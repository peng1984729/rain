# Part 1-任务三

![3](.\3.png)

```py
import torch.nn as nn

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        channels = 64
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(channels, channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        return self.relu(x + self.body(x))
```

**思考：**

`3x3` 的卷积核感受野有限，在不改变分辨率（因为是残差学习，无雨图 = 原图 - 雨，所以必须保证分辨率相同）的情况下，只能不断加深网络层数，但感受野线性增长，这需要堆叠非常深的网络，会导致过拟合、梯度消失、网络退化。

通过观察我们发现，图片中的雨丝形状多为细长条，所以我们可以适当改变卷积核大小（比如改为`3x10`），并且在随层数增大的过程中卷积核大小也适当增大，一次在有限层数中获得尽可能大的感受野。



```py
python main.py --data_dir "你的数据集地址" --model baseline --mode train --epochs 100
```

这个代码在终端中运行，`data_dir`中填写数据集地址即去雨训练用的图片，`model`选择模型，`mode`选择训练集或测试集，`epochs`表示训练轮数。

![4](.\4.png)

![5](.\5.png)