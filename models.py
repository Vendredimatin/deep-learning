from torchvision import models
import torch.nn as nn


def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes):
    ## your code here
    my_model = Net()

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
