#!/usr/bin/env python3
import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import ModelBetterCNN


# ---------- Utils ----------
def parse_window_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def normalize_mnist_like(x01: torch.Tensor) -> torch.Tensor:
    # igual ao teu dataset.py (MNIST normalize)
    return (x01 - 0.1307) / 0.3081


def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * (probs + eps).log()).sum(dim=1)


def inter_area_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    return iw * ih


def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = inter_area_xywh(a, b)
    if inter == 0:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    union = aw * ah + bw * bh - inter
    return float(inter / max(1, union))


def overlap_min_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    # interseção / área da menor bbox  -> bom para caixas contidas/quase iguais
    inter = inter_area_xywh(a, b)
    if inter == 0:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area_a = aw * ah
    area_b = bw * bh
    return float(inter / max(1, min(area_a, area_b)))


def nms_xywh_strong(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_thr: float,
    contain_thr: float
) -> List[int]:
    """
    Mantém as melhores caixas e remove:
      - caixas com IoU alto
      - caixas "quase iguais/contidas" via overlap_min (inter / area menor)
    """
    if not boxes:
        return []

    idxs = np.argsort(-np.array(scores))  # desc
    keep: List[int] = []

    while len(idxs) > 0:
        cur = int(idxs[0])
        keep.append(cur)

        rest = idxs[1:]
        new_rest = []
        for j in rest:
            j = int(j)
            iou = iou_xywh(boxes[cur], boxes[j])
            contain = overlap_min_xywh(boxes[cur], boxes[j])

            # remove se muito sobreposta OU se uma cobre muito a outra
            if (iou > iou_thr) or (contain > contain_thr):
                continue
            new_rest.append(j)

        idxs = np.array(new_rest, dtype=int)

    return keep


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_dir", type=str, required=True,
                        help="ex: ./data_scenes/B")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--weights", type=str, required=True,
                        help="Caminho para model.pth da T1")
    parser.add_argument("--out_dir", type=str, default="./experiments/tarefa3_sliding_v2")

    # varrimento
    parser.add_argument("--max_images", type=int, default=25)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--window_sizes", type=str, default="28")
    parser.add_argument("--batch_windows", type=int, default=256)

    # filtros
    parser.add_argument("--conf_thr", type=float, default=0.995)
    parser.add_argument("--entropy_thr", type=float, default=1.0)
    parser.add_argument("--min_maxpix", type=int, default=20,
                        help="se crop.max() < isto, ignora (fundo)")
    parser.add_argument("--min_box", type=int, default=0,
                        help="ignora bboxes com w/h < min_box (0 desliga)")

    # NMS forte
    parser.add_argument("--nms_iou", type=float, default=0.15)
    parser.add_argument("--contain_thr", type=float, default=0.6)

    # pós-processamento
    parser.add_argument("--keep_topk", type=int, default=0,
                        help="se >0, mantém só top-K por score (0 desliga)")
    parser.add_argument("--assume_single_digit", action="store_true",
                        help="para datasets A/B: mantém só a melhor deteção por imagem")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelBetterCNN().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- list images ----
    img_dir = os.path.join(args.scene_dir, args.split, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Não encontrei: {img_dir}")

    all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
    all_imgs = all_imgs[: max(1, min(args.max_images, len(all_imgs)))]

    win_sizes = parse_window_sizes(args.window_sizes)

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for fname in tqdm(all_imgs, desc="Sliding window v2"):
        path = os.path.join(img_dir, fname)
        im = Image.open(path).convert("L")
        arr = np.array(im, dtype=np.uint8)
        H, W = arr.shape

        boxes: List[Tuple[int, int, int, int]] = []
        labels: List[int] = []
        scores: List[float] = []

        for ws in win_sizes:
            if ws > H or ws > W:
                continue

            batch_patches: List[torch.Tensor] = []
            batch_boxes: List[Tuple[int, int, int, int]] = []

            for y in range(0, H - ws + 1, args.stride):
                for x in range(0, W - ws + 1, args.stride):
                    crop = arr[y:y + ws, x:x + ws]

                    # filtro rápido de fundo
                    if int(crop.max()) < args.min_maxpix:
                        continue

                    # (opcional) filtrar por tamanho mínimo
                    if args.min_box > 0 and ws < args.min_box:
                        continue

                    t = torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,ws,ws)

                    # resize para 28x28 (classificador)
                    if ws != 28:
                        t = F.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)

                    t = normalize_mnist_like(t)

                    batch_patches.append(t)
                    batch_boxes.append((x, y, ws, ws))

                    if len(batch_patches) >= args.batch_windows:
                        X = torch.cat(batch_patches, dim=0).to(device)
                        with torch.no_grad():
                            logits = model(X)
                            probs = torch.softmax(logits, dim=1)
                            conf, pred = probs.max(dim=1)
                            ent = softmax_entropy(probs)

                        conf = conf.cpu().numpy()
                        pred = pred.cpu().numpy()
                        ent = ent.cpu().numpy()

                        for bb, c, p, e in zip(batch_boxes, conf, pred, ent):
                            if (c >= args.conf_thr) and (e <= args.entropy_thr):
                                boxes.append(bb)
                                labels.append(int(p))
                                scores.append(float(c))

                        batch_patches = []
                        batch_boxes = []

            # último batch
            if batch_patches:
                X = torch.cat(batch_patches, dim=0).to(device)
                with torch.no_grad():
                    logits = model(X)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    ent = softmax_entropy(probs)

                conf = conf.cpu().numpy()
                pred = pred.cpu().numpy()
                ent = ent.cpu().numpy()

                for bb, c, p, e in zip(batch_boxes, conf, pred, ent):
                    if (c >= args.conf_thr) and (e <= args.entropy_thr):
                        boxes.append(bb)
                        labels.append(int(p))
                        scores.append(float(c))

        # ---------- NMS FORTE ----------
        keep = nms_xywh_strong(boxes, scores, iou_thr=args.nms_iou, contain_thr=args.contain_thr)
        boxes  = [boxes[i] for i in keep]
        labels = [labels[i] for i in keep]
        scores = [scores[i] for i in keep]

        # ---------- keep top-K (opcional) ----------
        if args.keep_topk and len(scores) > args.keep_topk:
            order = np.argsort(-np.array(scores))[:args.keep_topk]
            boxes  = [boxes[int(i)] for i in order]
            labels = [labels[int(i)] for i in order]
            scores = [scores[int(i)] for i in order]

        # ---------- single-digit mode (A/B) ----------
        if args.assume_single_digit and len(scores) > 0:
            best = int(np.argmax(np.array(scores)))
            boxes, labels, scores = [boxes[best]], [labels[best]], [scores[best]]

        # desenhar deteções
        out_img = im.convert("RGB")
        draw = ImageDraw.Draw(out_img)
        for (x, y, w, h), lab, sc in zip(boxes, labels, scores):
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
            draw.text((x, max(0, y - 10)), f"{lab} {sc:.2f}", fill=(255, 0, 0))

        out_path = os.path.join(args.out_dir, f"det_{os.path.splitext(fname)[0]}.png")
        out_img.save(out_path)

        results.append({
            "file": fname,
            "detections": [{"bbox": list(b), "label": int(l), "score": float(s)}
                           for b, l, s in zip(boxes, labels, scores)]
        })

    out_json = os.path.join(args.out_dir, "detections.json")
    with open(out_json, "w") as f:
        json.dump({
            "scene_dir": args.scene_dir,
            "split": args.split,
            "params": vars(args),
            "results": results
        }, f, indent=2)

    dt = time.time() - t0
    print("\n✅ Done")
    print("Output dir:", args.out_dir)
    print("detections.json:", out_json)
    print(f"Tempo total: {dt:.2f}s | {dt / max(1, len(all_imgs)):.2f}s por imagem")


if __name__ == "__main__":
    main()
