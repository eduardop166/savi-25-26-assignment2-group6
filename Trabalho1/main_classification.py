#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

from dataset import Dataset
from model import ModelBetterCNN
from trainer import Trainer
import torch, os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default="./data_mnist",
                        help="Pasta onde o torchvision vai guardar/ler o MNIST")
    parser.add_argument("--percentage_examples", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--experiment_path", type=str, default="./experiments/tarefa1")

    args = vars(parser.parse_args())

    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args["experiment_full_name"] = os.path.join(args["experiment_path"], exp_name)
    os.makedirs(args["experiment_full_name"], exist_ok=True)

    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)

    print("Experiment:", args["experiment_full_name"])
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    model = ModelBetterCNN()
    trainer = Trainer(args, train_dataset, test_dataset, model)

    trainer.train()
    torch.save(trainer.model.state_dict(), os.path.join(trainer.args["experiment_full_name"], "model.pth"))
    print("Saved model.pth")
    trainer.evaluate()






if __name__ == "__main__":
    main()
