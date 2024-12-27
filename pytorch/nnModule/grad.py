import torch
from graphviz import Digraph
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(2, 2, requires_grad=True)
x2 = torch.randn(2, 2, requires_grad=True)

# y, z = x.chunk(2, dim=0)

y = x.matmul(x2)

z = y * 2 + x

# z.sum().backward()
print(z)
# import ipdb
# ipdb.set_trace()
# print(x.grad)

# class GradFn:
#     def grad_fn


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

# 实例化模型
model = SimpleMNISTModel()
x = torch.randn(2, 28 * 28).requires_grad_()
mnist_out = model(x)

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

# 示例
model = resnet18(num_classes=1000)

image = torch.randn(1, 3, 224, 224).requires_grad_()

out = model(image)



node_id = 0


def add_node(G: Digraph, input_grad_fn, gradfn_to_viznode:dict,):
    if input_grad_fn in gradfn_to_viznode:
        return
    global node_id
    gradfn_to_viznode[input_grad_fn] = str(node_id)
    node_id += 1
    print(f"adding {str(type(input_grad_fn))=}")
    if type(input_grad_fn).__name__ == "AccumulateGrad":
        v = input_grad_fn.variable
        G.node(gradfn_to_viznode[input_grad_fn], label=f"AccumulateGrad:{list(v.shape)},{str(v.dtype)}", shape='rectangle', style='filled',)
    else:
        G.node(gradfn_to_viznode[input_grad_fn], label=f"{input_grad_fn.name()}")
    


def _visual_graph(G: Digraph, grad_fn, gradfn_to_viznode: dict, visited: set):
    
    if grad_fn is None:
        print(f"grad_fn is None")
        return
    
    if grad_fn in visited:
        return
    visited.add(grad_fn)
    
    for (input_grad_fn, index) in grad_fn.next_functions:
        # 创建节点
        #input_grad_fn 有3种情况：
        # 1. AccumulateGrad
        # 2. grad_fn比如AddBackward0
        # 3. None
        if input_grad_fn is None:
            continue
        
        add_node(G, input_grad_fn, gradfn_to_viznode)
        
        #创建边
        G.edge(head_name=gradfn_to_viznode[grad_fn], tail_name=gradfn_to_viznode[input_grad_fn])
        
        #如果不是AccumulateGrad，则递归处理
        if type(input_grad_fn).__name__ == "AccumulateGrad":       
            continue
        _visual_graph(G, input_grad_fn, gradfn_to_viznode, visited)


def visual_graph(t: torch.Tensor):
    G = Digraph()
    if t.grad_fn is not None:
        global node_id
        G.node("out", label=f"out:{list(t.shape)},{str(t.dtype)}", shape='rectangle', style='filled', )
        gradfn_to_viznode = {}
        add_node(G, t.grad_fn, gradfn_to_viznode)
        G.edge(head_name="out", tail_name=gradfn_to_viznode[t.grad_fn])
        _visual_graph(G, t.grad_fn, gradfn_to_viznode, set())
    else:
        assert t.grad_fn is not None, f"{t=} should have a grad_fn"

    G.render("graph", format='png', view=False)

visual_graph(out)