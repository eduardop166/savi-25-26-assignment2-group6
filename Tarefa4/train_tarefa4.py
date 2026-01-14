import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from model_tarefa4 import ModelIntegrated
import os

class DetectionDataset(Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST(root='./data_mnist', train=train, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.mnist) * 2 # 50% dígitos, 50% fundo puro

    def __getitem__(self, idx):
        canvas = np.zeros((64, 64), dtype=np.uint8)
        
        if idx % 2 == 0: # Caso com Dígito
            img, label = self.mnist[idx // 2]
            img = np.array(img)
            # Posição aleatória (Lógica Tarefa 2)
            x, y = np.random.randint(0, 36), np.random.randint(0, 36)
            canvas[y:y+28, x:x+28] = img
            # BBox normalizada (x, y, w, h)
            bbox = torch.tensor([x/64, y/64, 28/64, 28/64], dtype=torch.float)
            return self.transform(canvas), label, bbox
        else: # Caso de Fundo (Classe 10)
            label = 10
            bbox = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float)
            return self.transform(canvas), label, bbox

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelIntegrated().to(device)
    loader = DataLoader(DetectionDataset(train=True), batch_size=64, shuffle=True)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Iniciando Treino Integrado na {device}...")
    for epoch in range(2): # 5 épocas são suficientes para convergência inicial
        model.train()
        for imgs, labels, gt_bboxes in tqdm(loader):
            imgs, labels, gt_bboxes = imgs.to(device), labels.to(device), gt_bboxes.to(device)
            
            optimizer.zero_grad()
            pred_cls, pred_bboxes = model(imgs)
            
            loss_cls = criterion_cls(pred_cls, labels)
            # A perda de regressão só conta para quando existe um dígito (label < 10)
            mask = (labels < 10).float().unsqueeze(1)
            loss_bbox = criterion_bbox(pred_bboxes * mask, gt_bboxes * mask)
            
            loss = loss_cls + (loss_bbox * 5.0) # Peso maior na regressão para forçar precisão
            loss.backward()
            optimizer.step()
            
    os.makedirs("experiments/tarefa4", exist_ok=True)
    torch.save(model.state_dict(), "experiments/tarefa4/model_integrated.pth")
    print("Modelo salvo com sucesso!")

if __name__ == "__main__":
    train()