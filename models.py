import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    ile = model.fc.in_features
    model.fc = nn.Linear(ile, num_classes)
    return model


def build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    ile = model.fc.in_features
    model.fc = nn.Linear(ile, num_classes)
    return model


def build_mobilenet(num_classes):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    ile = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(ile, num_classes)
    return model
