# Projeto (Tarefa 1 + 2 + 3) — guia mega rápido

## 0) Instalar cenas (uma vez só)
Na raiz do repo (onde estão as pastas `tarefa1/`, `tarefa2/`, `tarefa3/`):

    pip install -r requirements.txt

---

# ✅ Tarefa 1 — Classificação MNIST
Entra:

    cd tarefa1

Corre (exemplo 10 épocas):

    python3 main_classification.py --num_epochs 10

O que isto faz:
- treina o modelo para reconhecer dígitos 0–9
- no fim faz avaliação e guarda:
  - gráficos (loss/accuracy)
  - matriz de confusão
  - métricas (precision/recall/f1)
  - `model.pth` (pesos do modelo) ✅ vais precisar disto na T3

Outputs:
- `./experiments/tarefa1/<timestamp>/`

### Ficheiros (T1)
- `main_classification.py`: arranca tudo (treino + avaliação + guarda outputs)
- `dataset.py`: carrega MNIST e normaliza
- `model.py`: define o modelo (`ModelBetterCNN`)
- `trainer.py`: faz treino, gráficos, matriz confusão e métricas

---

# ✅ Tarefa 2 — Gerar datasets A/B/C/D + stats
Entra:

    cd ../tarefa2

## Gerar dataset (exemplo: D)
Começa pequeno:

    python3 main_generate_dataset.py --version D --out_dir ../data_scenes/D --canvas_size 128 --n_train 2000 --n_test 400

Opcional: stats + mosaico:

    python3 main_dataset_stats.py --dataset_dir ../data_scenes/D --split test

O que isto faz:
- `main_generate_dataset.py`: cria imagens grandes + `annotations.json` (bounding boxes)
- `main_dataset_stats.py`: mosaico com bboxes + histogramas + stats

### Ficheiros (T2)
- `main_generate_dataset.py`: gera as cenas e annotations
- `main_dataset_stats.py`: gera mosaicos + estatísticas

---

# ✅ Tarefa 3 — Sliding Window (deteção nas cenas)
Entra:

    cd ../tarefa3

## Correr sliding window (exemplo para dataset D)
ATENÇÃO: troca `<timestamp>` pelo nome da pasta criada na T1.

    python3 main_sliding_window_v2.py --scene_dir ../data_scenes/D --split test --weights ../tarefa1/experiments/tarefa1/<timestamp>/model.pth --out_dir ./experiments/tarefa3_D --max_images 25 --stride 6 --window_sizes 22,24,26,28,30,32,34,36 --conf_thr 0.995 --entropy_thr 1.0

Outputs:
- `./experiments/tarefa3_D/`
  - `det_*.png` (imagens com caixas)
  - `detections.json`

### Ficheiros (T3)
- `main_sliding_window_v2.py`: sliding window + filtros + NMS + outputs
- `model.py`: define o modelo para carregar o `model.pth`

---

## Dicas rápidas
- Lento? aumenta `--stride` (8 ou 10)
- Muito lixo? sobe `--conf_thr` (0.998) e baixa `--entropy_thr` (0.9)
- Para A/B (1 dígito): podes usar `--assume_single_digit` (fica só com 1 deteção)
