import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        
        if task == 'multi':
            self.classifer = nn.Linear(256, 3)
        else:
            self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output


class CustomModel(nn.Module):
    def __init__(self, task):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-2]
        self.pretrained_model = nn.Sequential(*modules)
        
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)

        if task == 'multi':
            self.classifier = nn.Linear(2048, 3)
        else:
            self.classifier = nn.Linear(2048, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output
