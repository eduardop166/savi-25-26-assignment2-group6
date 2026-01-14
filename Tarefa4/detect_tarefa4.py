#!/usr/bin/env python3
import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from torchvision.ops import nms  # Para eliminar caixas sobrepostas
from model_tarefa4 import ModelIntegrated 

def process_scene(model, img_path, device, transform, out_dir):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Passamos a imagem INTEIRA pela FCN (Vantagem da Tarefa 4)
    t = transform(img_gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # A rede devolve mapas de ativação [Batch, Canais, H', W']
        cls_logits, bbox_reg = model(t)
        probs = torch.softmax(cls_logits, dim=1)
        
        # Filtramos a classe 10 (fundo) e pegamos na melhor probabilidade entre 0-9
        conf, pred = probs[:, :10, :, :].max(1) 
        
        # 1. Threshold de Confiança (Reduz falsos positivos no fundo)
        mask = conf > 0.98 
        indices = torch.nonzero(mask[0]) 

    boxes, scores, labels = [], [], []
    win_size = 64
    stride = 8 # Definido pelos 3 MaxPools (2^3) do teu backbone

    for idx in indices:
        iy, ix = idx[0].item(), idx[1].item()
        
        # Coordenadas base do "campo recetivo" no mapa original
        start_x, start_y = ix * stride, iy * stride
        
        # Regressão (ajuste fino predito pela rede)
        dx, dy, dw, dh = bbox_reg[0, :, iy, ix].cpu().numpy()
        
        # Converter para coordenadas reais da imagem
        x1 = int(start_x + dx * win_size)
        y1 = int(start_y + dy * win_size)
        x2 = x1 + int(dw * win_size)
        y2 = y1 + int(dh * win_size)
        
        boxes.append([x1, y1, x2, y2])
        scores.append(conf[0, iy, ix].item())
        labels.append(pred[0, iy, ix].item())

    if boxes:
        # 2. Non-Maximum Suppression (Elimina as caixas repetidas no mesmo dígito)
        boxes_t = torch.tensor(boxes, dtype=torch.float)
        scores_t = torch.tensor(scores, dtype=torch.float)
        keep = nms(boxes_t, scores_t, iou_threshold=0.2) # Se sobrepor > 20%, mantém a melhor
        
        for k in keep:
            b = boxes[k]
            cv2.rectangle(img_bgr, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{labels[k]} ({scores[k]:.2f})", (b[0], b[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    save_path = os.path.join(out_dir, f"det_{os.path.basename(img_path)}")
    cv2.imwrite(save_path, img_bgr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--weights", type=str, default="experiments/tarefa4/model_integrated.pth")
    parser.add_argument("--out_dir", type=str, default="experiments/tarefa4/results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ModelIntegrated().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if os.path.isdir(args.scene):
        files = [os.path.join(args.scene, f) for f in os.listdir(args.scene) if f.lower().endswith(('.png', '.jpg'))]
        for f in files:
            process_scene(model, f, device, transform, args.out_dir)
    else:
        process_scene(model, args.scene, device, transform, args.out_dir)

    print(f"✅ Processamento concluído. Resultados em: {args.out_dir}")

if __name__ == "__main__":
    main()