#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def load_coco(path: str):
    with open(path, "r") as f:
        coco = json.load(f)
    # index annotations by image_id
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    return coco, ann_by_img


def draw_mosaic(images_dir: str, coco, ann_by_img, out_path: str, n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    imgs = coco["images"]
    n = min(n, len(imgs))
    sel = rng.choice(len(imgs), size=n, replace=False)

    grid = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(12, 12))

    for k, idx in enumerate(sel):
        info = imgs[idx]
        img_id = info["id"]
        img_path = os.path.join(images_dir, info["file_name"])  # file_name já inclui "images/xxx.png"
        im = np.array(Image.open(img_path))

        ax = plt.subplot(grid, grid, k + 1)
        ax.imshow(im, cmap="gray")
        ax.axis("off")

        # bboxes
        for a in ann_by_img[img_id]:
            x, y, w, h = a["bbox"]
            rect = Rectangle((x, y), w, h, fill=False, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, y, str(a["category_id"]), fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="ex: ./data_scenes/A")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out_dir", type=str, default="./experiments/tarefa2_stats")
    parser.add_argument("--mosaic_n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ann_path = os.path.join(args.dataset_dir, args.split, "annotations.json")
    images_dir = os.path.join(args.dataset_dir, args.split)  # tem subpasta images/

    coco, ann_by_img = load_coco(ann_path)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Stats básicas ----
    num_images = len(coco["images"])
    all_labels = []
    num_digits_per_img = []
    widths, heights, areas = [], [], []

    for info in coco["images"]:
        img_id = info["id"]
        anns = ann_by_img[img_id]
        num_digits_per_img.append(len(anns))
        for a in anns:
            all_labels.append(a["category_id"])
            x, y, w, h = a["bbox"]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    label_counts = Counter(all_labels)

    # guarda texto simples
    stats_txt = os.path.join(args.out_dir, f"stats_{os.path.basename(args.dataset_dir)}_{args.split}.txt")
    with open(stats_txt, "w") as f:
        f.write(f"Dataset: {args.dataset_dir} | split: {args.split}\n")
        f.write(f"Num images: {num_images}\n")
        f.write(f"Total digits (instances): {len(all_labels)}\n")
        f.write("\nDigits per image:\n")
        f.write(f"  mean={np.mean(num_digits_per_img):.3f}  std={np.std(num_digits_per_img):.3f}\n")
        f.write(f"  min={np.min(num_digits_per_img)}  max={np.max(num_digits_per_img)}\n")
        f.write("\nBBox size (w/h):\n")
        f.write(f"  mean_w={np.mean(widths):.2f} mean_h={np.mean(heights):.2f}\n")
        f.write(f"  min_w={np.min(widths)} max_w={np.max(widths)}\n")
        f.write(f"  min_h={np.min(heights)} max_h={np.max(heights)}\n")

        f.write("\nClass distribution:\n")
        for c in range(10):
            f.write(f"  {c}: {label_counts.get(c, 0)}\n")

    # ---- Gráficos ----
    # hist nº dígitos por imagem
    plt.figure()
    plt.title("Histogram: digits per image")
    plt.xlabel("#digits")
    plt.ylabel("count")
    plt.hist(num_digits_per_img, bins=np.arange(0, max(num_digits_per_img) + 2) - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"hist_digits_per_image_{os.path.basename(args.dataset_dir)}_{args.split}.png"))
    plt.close()

    # distribuição de classes
    plt.figure()
    plt.title("Class distribution")
    plt.xlabel("digit")
    plt.ylabel("count")
    xs = list(range(10))
    ys = [label_counts.get(i, 0) for i in xs]
    plt.bar(xs, ys)
    plt.xticks(xs, [str(i) for i in xs])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"class_distribution_{os.path.basename(args.dataset_dir)}_{args.split}.png"))
    plt.close()

    # tamanhos (w)
    plt.figure()
    plt.title("Histogram: bbox width")
    plt.xlabel("width (px)")
    plt.ylabel("count")
    plt.hist(widths, bins=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"hist_bbox_width_{os.path.basename(args.dataset_dir)}_{args.split}.png"))
    plt.close()

    # ---- Mosaico com bboxes ----
    mosaic_path = os.path.join(args.out_dir, f"mosaic_{os.path.basename(args.dataset_dir)}_{args.split}.png")
    draw_mosaic(images_dir, coco, ann_by_img, mosaic_path, n=args.mosaic_n, seed=args.seed)

    print("✅ Stats + imagens guardadas em:", args.out_dir)
    print(" -", os.path.basename(stats_txt))
    print(" - mosaic_*.png + hist_*.png + class_distribution_*.png")


if __name__ == "__main__":
    main()
