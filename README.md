# Projeto (Tarefa 1 + 2 + 3) — guia mega rápido

## 0) Instalar pacotes

ve os imports de cada ficheiro e faz pip install em todos os pacotes que nao tenhas

# Tarefa 1 — Classificação MNIST
Entra na pasta:

    cd tarefa1

Corre (exemplo 3 épocas):

    python3 main_classification.py --num_epochs 3

O que isto faz:
- treina o modelo para reconhecer dígitos 0–9
- no fim faz avaliação e guarda:
  - gráficos (loss/accuracy)
  - matriz de confusão
  - métricas (precision/recall/f1)
  - `model.pth` (pesos do modelo) vais precisar disto na Tarefa 3

Outputs:
- `./experiments/tarefa1/<timestamp>/`

### Ficheiros (T1)
- `main_classification.py`: arranca tudo (treino + avaliação + guarda outputs)
- `dataset.py`: carrega MNIST e normaliza
- `model.py`: define o modelo (`ModelBetterCNN`)
- `trainer.py`: faz treino, gráficos, matriz confusão e métricas

---

# Tarefa 2 — Gerar datasets A/B/C/D + stats
Entra na pasta:

    cd ../tarefa2

## Gerar dataset (exemplo: D) (há 4 opcoes, A,B,C e D)
Começa pequeno:

    python3 main_generate_dataset.py --version D --out_dir ../data_scenes/D --canvas_size 128 --n_train 2000 --n_test 400

Para gerar os resultados: stats + mosaico:

    python3 main_dataset_stats.py --dataset_dir ../data_scenes/D --split test

O que faz:
- `main_generate_dataset.py`: cria os datasets pedidos + `annotations.json` (bounding boxes)
- `main_dataset_stats.py`: estatisticas: mosaico com bboxes + histogramas + stats


---

# Tarefa 3 — Sliding Window (deteção nas cenas)
Entra:

    cd ../tarefa3

## Correr sliding window (exemplo para dataset D) 
ATENÇÃO: troca `<timestamp>` pelo nome da pasta criada na T1. (tens tambem que alterar o tipo de dataset que queres testar(A,B,C...))

    python3 main_sliding_window_v2.py --scene_dir ../data_scenes/D --split test --weights ../tarefa1/experiments/tarefa1/<timestamp>/model.pth --out_dir ./experiments/tarefa3_D --max_images 25 --stride 6 --window_sizes 22,24,26,28,30,32,34,36 --conf_thr 0.995 --entropy_thr 1.0

Outputs:
- `./experiments/tarefa3_D/`
  - `det_*.png` (imagens com os resultados)
  - `detections.json` 

### Ficheiros (T3)

- `main_sliding_window_v2.py`
  - É o “programa principal” da tarefa 3.
  - Abre as imagens das cenas (ex: `data_scenes/D/test/images/*.png`).
  - Varre cada imagem com **sliding window**: recorta muitos “quadradinhos” (janelas) em várias posições (e em várias escalas se passares vários `--window_sizes`).
  - Cada recorte é:
    - (se preciso) redimensionado para **28x28** (porque o classificador foi treinado em MNIST 28x28),
    - normalizado como o MNIST,
    - passado pelo modelo da Tarefa 1 para prever qual dígito é.
  - Depois aplica filtros para não aceitar tudo:
    - `--conf_thr`: só aceita se a probabilidade (softmax) for alta
    - `--entropy_thr`: só aceita se a previsão estiver “confiante” (entropia baixa)
    - `--min_maxpix`: ignora recortes quase pretos (fundo)
  - No fim faz **NMS** (Non-Maximum Suppression) para remover caixas repetidas/quase iguais (porque muitas janelas apanham o mesmo dígito).
  - Guarda:
    - `det_*.png` (imagens com as bounding boxes desenhadas e o dígito previsto)
    - `detections.json` (as deteções em formato fácil de ler/usar)

- `model.py`
  - Tem a definição da rede (ex: `ModelBetterCNN`) exatamente igual à usada na Tarefa 1.
  - É necessário porque o ficheiro `model.pth` guarda **só os pesos**, não guarda a “forma” do modelo.
  - Então o script faz:
    - cria o modelo (`model = ModelBetterCNN()`),
    - carrega os pesos (`model.load_state_dict(torch.load(model.pth))`),
    - usa o modelo para prever as janelas.

---

## COMENTARIOSSS!!

A tarefa 1 e 2 estao a funcionar bem, mas a 3 as vezes detecta objetos que nao estao la, falta corrigir isso. De resto funciona top
