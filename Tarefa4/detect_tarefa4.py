import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from model_tarefa4 import ModelIntegrated 

def process_scene_improved(model, img_path, device, transform, out_dir):
    # Abrir a imagem
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binarizar e achar os bonecos (objetos)
    _, img_bin = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    detections = []
    input_size = 64 
    half_size = input_size // 2

    # Borda preta pra não dar erro nos cantos ao recortar
    img_padded = cv2.copyMakeBorder(img_gray, half_size, half_size, half_size, half_size, 
                                    cv2.BORDER_CONSTANT, value=0)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i] # Centro do boneco
        
        # Ignorar coisas muito pequenas (ruído)
        if area < 20: continue 

        # Coordenadas no mapa com borda
        nx, ny = int(cx + half_size), int(cy + half_size)
        
        # Recorte de 64x64 centrado no boneco 
        crop = img_padded[ny - half_size : ny + half_size, 
                          nx - half_size : nx + half_size]

        # Mandar para a rede
        t = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_logits, _ = model(t)
            probs = torch.softmax(cls_logits, dim=1)
            conf, pred = probs.max(1)

            # Se for número (0-9) e tiver confiança mínima
            if pred < 20 and conf > 0.40:
                detections.append({
                    "bbox": [x, y, w, h],
                    "label": int(pred.item()),
                    "conf": float(conf.item())
                })

    # Desenhar os resultados
    for det in detections:
        x, y, w, h = det["bbox"]
        # Legenda com Número e Confiança (ex: 5 (0.98))
        text = f"{det['label']} ({det['conf']:.2f})"
        
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_bgr, text, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Guardar a imagem final
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"det_{os.path.basename(img_path)}"), img_bgr)

def main():
    # Ler argumentos do terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results_improved")
    args = parser.parse_args()

    # Configurar GPU ou CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo e pesos
    model = ModelIntegrated().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Normalização igual ao treino
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Se for pasta, corre tudo. Se for imagem, corre só uma.
    if os.path.isdir(args.scene):
        for f in os.listdir(args.scene):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_scene_improved(model, os.path.join(args.scene, f), device, transform, args.out_dir)
    else:
        process_scene_improved(model, args.scene, device, transform, args.out_dir)

if __name__ == "__main__":
    main()