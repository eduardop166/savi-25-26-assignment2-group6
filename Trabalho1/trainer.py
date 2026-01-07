import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


class Trainer:
    def __init__(self, args, train_dataset, test_dataset, model):
        self.args = args
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args["batch_size"],
            shuffle=True,
            num_workers=args.get("num_workers", 2),
            pin_memory=(self.device.type == "cuda"),
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=args.get("num_workers", 2),
            pin_memory=(self.device.type == "cuda"),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.get("lr", 1e-3))

        self.train_epoch_losses, self.test_epoch_losses = [], []
        self.train_epoch_accs, self.test_epoch_accs = [], []

        os.makedirs(self.args["experiment_full_name"], exist_ok=True)

    def train(self):
        print(f"Training on device: {self.device}")
        print("Train size:", len(self.train_dataloader.dataset))
        print("Test size:", len(self.test_dataloader.dataset))

        for epoch in range(self.args["num_epochs"]):
            # ---- TRAIN ----
            self.model.train()
            train_losses = []
            train_gts, train_preds = [], []

            for x, y in tqdm(self.train_dataloader, desc=f"Train {epoch+1}/{self.args['num_epochs']}"):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
                train_gts.extend(y.detach().cpu().tolist())

            train_loss = float(np.mean(train_losses))
            train_acc = float(accuracy_score(train_gts, train_preds))

            # ---- TEST ----
            self.model.eval()
            test_losses = []
            test_gts, test_preds = [], []

            with torch.no_grad():
                for x, y in tqdm(self.test_dataloader, desc=f"Test  {epoch+1}/{self.args['num_epochs']}"):
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                    test_losses.append(loss.item())
                    test_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    test_gts.extend(y.cpu().tolist())

            test_loss = float(np.mean(test_losses))
            test_acc = float(accuracy_score(test_gts, test_preds))

            self.train_epoch_losses.append(train_loss)
            self.test_epoch_losses.append(test_loss)
            self.train_epoch_accs.append(train_acc)
            self.test_epoch_accs.append(test_acc)

            print(f"\nEpoch {epoch+1}: train loss={train_loss:.4f} acc={train_acc:.4f} | test loss={test_loss:.4f} acc={test_acc:.4f}")
            self._draw_curves()

        print("Training completed.")

    def _draw_curves(self):
        # Loss
        plt.figure()
        plt.title("Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(range(1, len(self.train_epoch_losses) + 1), self.train_epoch_losses)
        plt.plot(range(1, len(self.test_epoch_losses) + 1), self.test_epoch_losses)
        plt.legend(["Train", "Test"])
        plt.tight_layout()
        plt.savefig(os.path.join(self.args["experiment_full_name"], "training_loss.png"))
        plt.close()

        # Accuracy
        plt.figure()
        plt.title("Accuracy vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.0])
        plt.plot(range(1, len(self.train_epoch_accs) + 1), self.train_epoch_accs)
        plt.plot(range(1, len(self.test_epoch_accs) + 1), self.test_epoch_accs)
        plt.legend(["Train", "Test"])
        plt.tight_layout()
        plt.savefig(os.path.join(self.args["experiment_full_name"], "training_accuracy.png"))
        plt.close()

    def evaluate(self):
        self.model.eval()
        gts, preds = [], []

        with torch.no_grad():
            for x, y in tqdm(self.test_dataloader, desc="Final evaluation"):
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                pred = torch.argmax(logits, dim=1)

                gts.extend(y.cpu().tolist())
                preds.extend(pred.cpu().tolist())

        cm = confusion_matrix(gts, preds, labels=list(range(10)))

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (MNIST)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(10), [str(i) for i in range(10)])
        plt.yticks(range(10), [str(i) for i in range(10)])

        for i in range(10):
            for j in range(10):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args["experiment_full_name"], "confusion_matrix.png"))
        plt.close()

        prec, rec, f1, support = precision_recall_fscore_support(
            gts, preds, labels=list(range(10)), average=None, zero_division=0
        )
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            gts, preds, average="macro", zero_division=0
        )
        acc = accuracy_score(gts, preds)

        metrics = {
            "accuracy": float(acc),
            "macro_avg": {"precision": float(macro_prec), "recall": float(macro_rec), "f1": float(macro_f1)},
            "per_class": {
                str(i): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
                for i in range(10)
            },
        }

        with open(os.path.join(self.args["experiment_full_name"], "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print("\nSaved: confusion_matrix.png + metrics.json")
        print("Accuracy:", acc, "| Macro F1:", macro_f1)
