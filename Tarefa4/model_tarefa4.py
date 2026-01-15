# model_tarefa4.py atualizado
import torch
import torch.nn as nn

class ModelIntegrated(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), # 64x64 -> 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=8), 
            nn.ReLU(),
            nn.Conv2d(256, 11, kernel_size=1) 
        )
        
        self.regressor = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        f = self.features(x)
        cls_map = self.classifier(f)
        bbox_map = self.regressor(f)
        
        # Ajuste para manter compatibilidade com o treino
        if x.shape[2] == 64 and x.shape[3] == 64:
            cls_map = cls_map.view(x.size(0), -1)
            bbox_map = bbox_map.view(x.size(0), -1)
            
        return cls_map, bbox_map