import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class XRayResNet50(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super(XRayResNet50, self).__init__()
        
        # 1. ResNet50'yi Yükle
        base_model = models.resnet50(weights=None)
        
        # 2. ResNet'in tüm katmanlarını bu sınıfın (self) bir parçası yap
        # (Böylece state_dict anahtarları 'conv1', 'layer1' diye başlar, 'backbone.conv1' diye değil)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        
        # 3. Son Katmanı (FC) Değiştir
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # Grad-CAM için buraya kanca atacağız

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x