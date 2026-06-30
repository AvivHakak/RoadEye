import torch
import torch.nn as nn
import torchvision.models as models

class RoadEyeGridModel(nn.Module):
    def __init__(self):
        super(RoadEyeGridModel, self).__init__()
        
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = mobilenet.features
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.detector = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.detector(features)
        return out