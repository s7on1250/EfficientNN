import math

import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)
        with torch.no_grad():
            for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
                fan_in = layer.weight[0].numel()
                layer.weight.uniform_(-math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
                layer.bias.zero_()

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

class CaffeBN(nn.Module):

    def __init__(self, channels, maf, ranks):
        super().__init__()
        self.maf = maf
        self.rank = 0
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('mean_sum', torch.zeros(ranks, channels))
        self.register_buffer('var_sum', torch.zeros(ranks, channels))
        self.register_buffer('count', torch.zeros(ranks))

    def forward(self, x):
        if self.training:
            mean = torch.zeros_like(self.weight)
            var = torch.zeros_like(self.weight)
            out = F.batch_norm(x, mean, var, None, None, True, 1.0, 1e-05)
            with torch.no_grad():
                self.mean_sum[self.rank].mul_(self.maf).add_(mean)
                self.var_sum[self.rank].mul_(self.maf).add_(var)
                self.count[self.rank].mul_(self.maf).add_(1.0)
        else:
            count = self.count[0].clamp_min(1e-12)
            out = F.batch_norm(x, self.mean_sum[0] / count, self.var_sum[0] / count, None, None, False, 0.0, 1e-05)
        return out * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class Block(nn.Module):

    def __init__(self, c, maf, ranks, down=False):
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(c, c, 3, 2 if down else 1, 1, bias=True)
        self.bn1 = CaffeBN(c, maf, ranks)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.bn2 = CaffeBN(c, maf, ranks)

    def forward(self, x):
        y = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        skip = F.avg_pool2d(x, 3, 2, ceil_mode=True) if self.down else x
        out = F.relu(y + skip)
        return torch.cat((out, torch.zeros_like(out)), dim=1) if self.down else out

class CifarResNet(nn.Module):

    def __init__(self, maf, ranks=8, depth=20):
        super().__init__()
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=True)
        self.bn = CaffeBN(16, maf, ranks)
        blocks = [Block(16, maf, ranks) for _ in range(n)]
        blocks += [Block(16, maf, ranks, down=True)]
        blocks += [Block(32, maf, ranks) for _ in range(n - 1)]
        blocks += [Block(32, maf, ranks, down=True)]
        blocks += [Block(64, maf, ranks) for _ in range(n - 1)]
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(64, 10)
        self.bn_layers = [m for m in self.modules() if isinstance(m, CaffeBN)]
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.normal_(0, math.sqrt(2 / m.weight[0].numel()))
                    m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.uniform_(-math.sqrt(3 / m.in_features), math.sqrt(3 / m.in_features))
                    m.bias.zero_()

    def rank(self, r):
        for m in self.bn_layers:
            m.rank = r

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc(F.avg_pool2d(x, 8).flatten(1))


def make_model(name, mode):
    if name == "lenet":
        return LeNet()
    return CifarResNet(0.95 if mode == "clr" else 0.999, ranks=8, depth=20)
