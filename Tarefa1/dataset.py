import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np


class Dataset(TorchDataset):
    def __init__(self, args, is_train: bool):
        self.args = args
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Faz download automático se faltar
        self.ds = MNIST(
            root=args["dataset_folder"],
            train=is_train,
            download=True,
            transform=self.transform
        )

        # Percentagem (1.0 = tudo)
        pct = float(args.get("percentage_examples", 1.0))
        pct = max(0.0, min(1.0, pct))

        if pct < 1.0:
            n = len(self.ds)
            k = max(1, int(round(n * pct)))
            rng = np.random.default_rng(seed=0)
            self.indices = rng.choice(n, size=k, replace=False).tolist()
        else:
            self.indices = None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.ds)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        img, label = self.ds[real_idx]  # label é int 0..9
        return img, torch.tensor(label, dtype=torch.long)
