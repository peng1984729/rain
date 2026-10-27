# Part 2-任务三

### 特征图中的L1：

**LPIPS的优势：**

原来的L1是将逐像素做差求和，找到像素点之间的差异，因此是局限的对’像素‘优化。

而VGG模型中，因为特征图中的点不再是单一像素，而有着更大感受野，涵盖了局部甚至全局的信息，因此我们将其做差再求和就可以对‘感知’进行优化。

**训练网络的作用：**

模型已经学会了基础的纹理细节，也可以做到识别物体，因此其特征图中的点涵盖了更多信息，它把“两张图像素是否接近”转换成“两张图在特征空间是否接近”。

```py
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features[:35]
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.meanm = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.l1 = nn.L1Loss()
    def forward(self, generated, ground_truth):
        generated = (generated - self.mean) / self.std
        ground_truth = (ground_truth - self.mean) / self.std
        gen_feat = self.vgg(generated)
        gt_feat = self.vgg(ground_truth)
        return self.l1(gen_feat, gt_feat)
```

#### 向量：

在程序设计中我认为向量就是数组，而向量的维度就是数组的大小即特征向量中的特征数。

#### 内积：

a，b两个n维向量。

两种表示方式：

$a · b = a_1b_1 + a_2b_2 + ... + a_nb_n$

$a · b = ||a|| × ||b|| × cosθ$，其中$||a|| = \sqrt{(a_1² + a_2² + ... + a_n²)}$

在线性代数中：

矩阵乘法中的行与列以此相乘再求和，就是求内积。

**内积的意义：**内积衡量两个向量“方向有多一致、彼此有多少重叠”。内积把每一维的乘积加起来，如果两个向量在某一维**同号**，这一维贡献正数，如果**异号**，贡献负数，如果某一维有一个是 0，这一维没有贡献。可以体现相关性。

余弦相似度是把长度影响去掉后，只看方向的一致程度。

### AdaptiveAvgPool2d：

自适应平均池化：不管输入特征图多大，都把它平均成你指定的输出尺寸。

自适应可以根据输入图像大小而调整池化窗口大小，从而保证输出尺寸不变。

### easy contrastive loss：

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class Easy_contrastive_loss(nn.Module):
    def __init__(self):
        super(Easy_contrastive_loss, self).__init__()
        self.vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features[:35]
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.222, 0.224, 0.225]).view(1, 3, 1, 1)

    def encode(self, x):
        x = (x - self.mean) / self.std
        feat = self.vgg(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        return F.normalize(feat, dim = 1)

    def forward(self, rainy, generated, ground_truth):
        rain_vec = self.encode(rainy)
        gt_vec = self.encode(ground_truth)
        gen_vec = self.encode(generated)
        cos_gt = F.cosine_embedding_loss(gen_vec, gt_vec, dim = 1)
        cos_rain = F.cosine_embedding_loss(gen_vec, rain_vec, dim = 1)
        loss = (1 - cos_gt + cos_rain).mean()
        return loss
```

### main中添加：

```py
    parser.add_argument("--use_contrastive_loss", action="store_true", help="Use easy contrastive loss")
    parser.add_argument("--lambda_contrastive", type=float, default=0.1, help="Weight for easy contrastive loss")
```

![7](.\7.png)