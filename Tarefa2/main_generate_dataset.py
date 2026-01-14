#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchvision.datasets import MNIST


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """a,b = (x,y,w,h). True se tiver interseção com área > 0."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if ax + aw <= bx or bx + bw <= ax:
        return False
    if ay + ah <= by or by + bh <= ay:
        return False
    return True


def pick_digit_size(version: str, base_size: int, scale_min: int, scale_max: int, rng: np.random.Generator) -> int:
    # Versões com escala: B e D (e.g. 22..36). Versões sem escala: A e C (fixo 28 por default).
    if version in ["B", "D"]:
        return int(rng.integers(scale_min, scale_max + 1))
    return base_size


def pick_num_digits(version: str, rng: np.random.Generator, multi_min: int, multi_max: int) -> int:
    # Versões multi: C e D (e.g. 3..5). Versões single: A e B (1).
    if version in ["C", "D"]:
        return int(rng.integers(multi_min, multi_max + 1))
    return 1


def generate_one_scene(
    mnist: MNIST,
    canvas_size: int,
    version: str,
    base_size: int,
    scale_min: int,
    scale_max: int,
    multi_min: int,
    multi_max: int,
    max_tries_per_digit: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Returns:
      canvas (H,W) uint8
      annotations: list of {"bbox":[x,y,w,h], "category_id":int}
    """
    H = W = canvas_size

    # Se falhar a colocar (por falta de espaço), recomeça a cena
    for _restart in range(50):
        canvas = np.zeros((H, W), dtype=np.uint8)
        annots: List[Dict[str, Any]] = []
        placed_boxes: List[Tuple[int, int, int, int]] = []

        n_digits = pick_num_digits(version, rng, multi_min, multi_max)

        ok = True
        for _k in range(n_digits):
            # escolhe um exemplo MNIST ao acaso
            idx = int(rng.integers(0, len(mnist)))
            digit_img_pil, digit_label = mnist[idx]  # PIL + int
            size = pick_digit_size(version, base_size, scale_min, scale_max, rng)

            # redimensiona
            digit_img_pil = digit_img_pil.resize((size, size), resample=Image.Resampling.BILINEAR)
            digit_arr = np.array(digit_img_pil, dtype=np.uint8)  # (size,size), 0..255

            # tenta posicionar sem overlap
            placed = False
            for _try in range(max_tries_per_digit):
                x = int(rng.integers(0, W - size + 1))
                y = int(rng.integers(0, H - size + 1))
                new_box = (x, y, size, size)

                if any(boxes_intersect(new_box, old) for old in placed_boxes):
                    continue

                # cola no canvas (sem overlap, pode ser assignment; uso maximum por segurança)
                roi = canvas[y:y+size, x:x+size]
                canvas[y:y+size, x:x+size] = np.maximum(roi, digit_arr)

                placed_boxes.append(new_box)
                annots.append({"bbox": [x, y, size, size], "category_id": int(digit_label)})
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return canvas, annots

    raise RuntimeError("Não consegui gerar uma cena sem sobreposição (tenta aumentar canvas ou reduzir nº dígitos).")


def make_coco(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"images": [], "annotations": [], "categories": categories}


def main():
    parser = argparse.ArgumentParser()

    # output / versões
    parser.add_argument("--version", type=str, default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--out_dir", type=str, default="./data_scenes/versionA")

    # tamanhos / quantidades
    parser.add_argument("--canvas_size", type=int, default=128, help="ex: 100, 128")
    parser.add_argument("--n_train", type=int, default=60000)
    parser.add_argument("--n_test", type=int, default=10000)

    # regras do enunciado
    parser.add_argument("--avoid_overlap", action="store_true", default=True)
    parser.add_argument("--scale_min", type=int, default=22)
    parser.add_argument("--scale_max", type=int, default=36)
    parser.add_argument("--base_size", type=int, default=28, help="tamanho fixo para A/C (sem escala)")
    parser.add_argument("--multi_min", type=int, default=3)
    parser.add_argument("--multi_max", type=int, default=5)

    # controlo
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tries_per_digit", type=int, default=200)

    # MNIST root (só para cache/download)
    parser.add_argument("--mnist_root", type=str, default="./data_mnist")

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # MNIST sem transform (PIL)
    mnist_train = MNIST(root=args.mnist_root, train=True, download=True)
    mnist_test = MNIST(root=args.mnist_root, train=False, download=True)

    # pastas
    train_img_dir = os.path.join(args.out_dir, "train", "images")
    test_img_dir = os.path.join(args.out_dir, "test", "images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    categories = [{"id": i, "name": str(i)} for i in range(10)]

    # ---- TRAIN ----
    coco_train = make_coco(categories)
    ann_id = 1
    img_id = 1
    print(f"Gerar TRAIN: {args.n_train} imagens (versão {args.version}) em {train_img_dir}")

    for i in tqdm(range(args.n_train)):
        canvas, annots = generate_one_scene(
            mnist=mnist_train,
            canvas_size=args.canvas_size,
            version=args.version,
            base_size=args.base_size,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            multi_min=args.multi_min,
            multi_max=args.multi_max,
            max_tries_per_digit=args.max_tries_per_digit,
            rng=rng,
        )

        file_name = f"{i:06d}.png"
        Image.fromarray(canvas).save(os.path.join(train_img_dir, file_name))

        coco_train["images"].append({
            "id": img_id,
            "file_name": os.path.join("images", file_name),
            "width": args.canvas_size,
            "height": args.canvas_size
        })

        for a in annots:
            coco_train["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": a["bbox"],  # [x,y,w,h]
                "category_id": a["category_id"]
            })
            ann_id += 1

        img_id += 1

    with open(os.path.join(args.out_dir, "train", "annotations.json"), "w") as f:
        json.dump(coco_train, f, indent=2)

    # ---- TEST ----
    coco_test = make_coco(categories)
    ann_id = 1
    img_id = 1
    print(f"Gerar TEST: {args.n_test} imagens (versão {args.version}) em {test_img_dir}")

    for i in tqdm(range(args.n_test)):
        canvas, annots = generate_one_scene(
            mnist=mnist_test,
            canvas_size=args.canvas_size,
            version=args.version,
            base_size=args.base_size,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            multi_min=args.multi_min,
            multi_max=args.multi_max,
            max_tries_per_digit=args.max_tries_per_digit,
            rng=rng,
        )

        file_name = f"{i:06d}.png"
        Image.fromarray(canvas).save(os.path.join(test_img_dir, file_name))

        coco_test["images"].append({
            "id": img_id,
            "file_name": os.path.join("images", file_name),
            "width": args.canvas_size,
            "height": args.canvas_size
        })

        for a in annots:
            coco_test["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": a["bbox"],
                "category_id": a["category_id"]
            })
            ann_id += 1

        img_id += 1

    with open(os.path.join(args.out_dir, "test", "annotations.json"), "w") as f:
        json.dump(coco_test, f, indent=2)

    print("\n✅ Dataset gerado em:", args.out_dir)
    print("   - train/images/*.png + train/annotations.json")
    print("   - test/images/*.png  + test/annotations.json")


if __name__ == "__main__":
    main()