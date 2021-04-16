from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes):
    ## your code here
    #my_model = Net()
    my_model = ResNet(BasicBlock, [2,2,2,2])

    return my_model


def model_C(num_classes):
    ## your code here
    return


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attentionLayer1 = AttentionLayer(channel=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.attentionLayer2 = AttentionLayer(channel=32)

        self.fc = nn.Linear(100352, 10)

    def forward(self,x):
        out = self.conv1(x)
        out = self.attentionLayer1(out)
        out = self.conv2(out)
        out = self.attentionLayer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, channel, reduce = 16):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduce, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduce , channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        batch, channel, _, _ = input.size()
        output = self.avg_pool(input).view(batch, channel)
        output = self.fc(output).view(batch, channel, 1, 1)
        return input * output.expand_as(input)

# resnet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, out_features, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride=, padding=1, bias=False)
        # 待处理数据的通道数
        self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features,out_features,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_features != self.expansion * out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features,self.expansion*out_features,1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(self.expansion * out_features)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()

        self.in_features = 64
        self.conv1 = nn.Conv2d(3, self.in_features, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_features)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    
    def _make_layer(self,block,features, num_blocks,stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_features, frozenset, stride))
            self.in_features = features * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
