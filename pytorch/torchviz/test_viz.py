import sys
sys.path.append(".")
from viz import viz_graph

import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)      # 隐藏层
        self.fc3 = nn.Linear(64, 10)       # 输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

# 定义 ResNet18
class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 创建 ResNet-18 模型实例
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


if __name__ == "__main__":
    mnist = SimpleMNISTModel()
    x = torch.randn(2, 28 * 28).requires_grad_()
    mnist_out = mnist(x)
    viz_graph(mnist_out).render("mnist", format="png")

    resnet = resnet18(num_classes=1000)
    image = torch.randn(1, 3, 224, 224).requires_grad_()
    resnet_out = resnet(image)
    viz_graph(resnet_out).render("resnet18", format="png")
    
    #index=1
    a = torch.randn(10, 2).requires_grad_()
    x, y = a.chunk(2)
    z = x + y * 2
    viz_graph(z).render("chunk", format="png")
    
    a = torch.randn(10, 2).requires_grad_()
    b = a + 2 #如何根据dx dy dz计算db?
    x = b * 2
    y = b / 2
    z = b.pow(2)
    out = (x + y + z).abs()
    viz_graph(out).render("test", format="png")
    