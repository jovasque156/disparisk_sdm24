import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                    padding=padding, 
                                    stride=stride, 
                                    bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels)
        # by default, stride=1
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.relu(out)
        return out

class resnet18(nn.Module):
    '''
    A Residual network Resnet18
    '''
    def __init__(self, block:Type[ResidualBlock], num_classes=10):
        super(resnet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)        
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))        
        self.fc = nn.Linear(514, num_classes)
        
    def _make_layer(self, block: Type[ResidualBlock], channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x, sens_attr):
        # ipdb.set_trace()
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        sens_attr = torch.flatten(sens_attr, 1)
        
        out = torch.cat((out, sens_attr), dim=1)
        
        out = self.fc(out)
        
        return out

# Based on https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6 + 2, 4096), # 2 because of the sensitive attribute
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, sens_attr) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, sens_attr), dim=1)
        out = self.classifier(out)
        return out
    
class OneMLP(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(OneMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(224*224*3+2, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x, sens_attr) -> torch.Tensor:
        out = torch.flatten(x,1)
        out = torch.cat((out, sens_attr), dim=1)
        out = self.classifier(out)
        return out

class TwoMLP(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(TwoMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(224*224*3+2, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x, sens_attr) -> torch.Tensor:
        out = torch.flatten(x, 1)
        out = torch.cat((out, sens_attr), dim=1)
        out = self.classifier(out)
        return out
    
class Linear(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super(Linear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(224*224*3+2, num_classes),
        )

    def forward(self, x, sens_attr) -> torch.Tensor:
        out = torch.flatten(x, 1)
        out = torch.cat((out, sens_attr), dim=1)
        out = self.classifier(out)
        return out