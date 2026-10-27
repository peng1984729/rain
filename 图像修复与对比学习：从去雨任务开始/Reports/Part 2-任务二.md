# Part 2-任务二

```py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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


class DerainNet(nn.Module):
    def __init__(self, channels = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
        )
        self.branch_full = nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.branch_half = nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.branch_quarter = nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True),
        )
        self.head = nn.Conv2d(channels, 3, 1)

    def forward(self, x):
        feat = self.stem(x)
        half_in = self.pool(feat)
        quarter_in = self.pool(half_in)

        branches = (self.branch_full, self.branch_half, self.branch_quarter)
        inputs = (feat, half_in, quarter_in)
        features = []
        for branch, inp in zip(branches, inputs):
            out = torch.relu(inp + branch(inp))
            features.append(
                F.interpolate(out, size = feat.shape[-2:], mode = "bilinear", align_corners = False)
            )

        fused = self.fuse(torch.cat(features, dim = 1))
        rain = torch.relu(self.head(fused))
        return x - rain

```

stem做一次，把 RGB 的 3 个通道投影到 channel个特征通道，三个分支分别工作在 原图、1/2、1/4三种分辨率上，每个分支都是 4 个 ResBlock = 8 层 3×3 卷积，每个分支内部先做一次“处理前特征 + 处理后特征”的残差相加，再上采样回原图尺寸，三个分支在通道维拼接成 3channel 通道，用卷积融合，最后输出 3 通道雨层，最终`去雨图 = 输入 - 雨层`得到。

### 原因：

我们使用先下采样再上采样的方式可以用相对小的损失增大感受野，不再需要堆叠网络层数。同时我们也可以处理不同尺寸、疏密程度的雨丝。

### 下采样：

就是把特征图的空间尺寸变小，增大感受野。

例如：`128×128 → 64×64 → 32×32`

本题中我们使用池化操作来实现：`self.pool = nn.MaxPool2d(kernel_size=2, stride=2)`

### 上采样：

同理就是把特征图的空间尺寸变大。

例如：`32×32 → 64×64 → 128×128`

本题中我们使用双线性插值来实现：

```python
F.interpolate(half, size=feat.shape[-2:], mode="bilinear", align_corners=False)
```

双线性插值的意思是：输出像素的值，由它周围最近的 4 个输入像素按距离加权平均得到，离哪个像素近，哪个像素的权重就大。

### Concat：

concat就是拼接。

```python
torch.cat([full, half, quarter], dim=1)
```

三个分支的形状都是 `[B, C, H, W]`，沿通道维 `dim=1` 拼接后变成：

```py
[B, C, H, W] × 3  ──Concat(dim = 1)──→  [B, 3C, H, W]
```
