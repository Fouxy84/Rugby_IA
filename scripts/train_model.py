"""
Script d'entraînement — Phase Classifier (CNN-LSTM)

Usage :
    python scripts/train_model.py --data-dir data/annotations --epochs 50

Structure attendue du répertoire d'annotations :
    data/annotations/
        jeu_courant/  (clips de N frames)
        melee/
        touche/
        essai/
        ...
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.analysis.phase_classifier import PhaseClassifierModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_model")


def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RugbyPhaseDataset(Dataset):
    """
    Charge des clips vidéo annotés par phase de jeu.
    Chaque clip est un répertoire de frames JPEG numérotées.
    """

    def __init__(self, clips: list[tuple[Path, int]], seq_len: int = 30):
        self.clips = clips
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir, label = self.clips[idx]
        frames_paths = sorted(clip_dir.glob("*.jpg"))

        # Sous-échantillonnage ou padding pour atteindre seq_len
        if len(frames_paths) >= self.seq_len:
            indices = np.linspace(0, len(frames_paths) - 1, self.seq_len, dtype=int)
            frames_paths = [frames_paths[i] for i in indices]
        else:
            # Répétition de la dernière frame
            while len(frames_paths) < self.seq_len:
                frames_paths.append(frames_paths[-1])

        tensors = []
        for fp in frames_paths:
            img = cv2.imread(str(fp))
            if img is None:
                img = np.zeros((112, 112, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))

        return torch.stack(tensors), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train(args):
    cfg = load_config()
    phase_cfg = cfg["phase_classifier"]
    classes = phase_cfg["classes"]
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # --- Collecte des clips ---
    data_dir = Path(args.data_dir)
    all_clips = []
    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir() and cls_dir.name in class_to_idx:
            label = class_to_idx[cls_dir.name]
            for clip_dir in cls_dir.iterdir():
                if clip_dir.is_dir():
                    all_clips.append((clip_dir, label))

    if not all_clips:
        logger.error("Aucun clip trouvé dans %s", data_dir)
        sys.exit(1)

    logger.info("Clips trouvés : %d", len(all_clips))

    train_clips, val_clips = train_test_split(
        all_clips, test_size=0.2, random_state=42,
        stratify=[c[1] for c in all_clips],
    )

    seq_len = phase_cfg["sequence_length"]
    train_ds = RugbyPhaseDataset(train_clips, seq_len=seq_len)
    val_ds   = RugbyPhaseDataset(val_clips,   seq_len=seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Modèle ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseClassifierModel(n_classes=n_classes, seq_len=seq_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # --- MLflow ---
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seq_len": seq_len,
            "n_classes": n_classes,
        })

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for X, y in train_dl:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            correct = total = 0
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_dl:
                    X, y = X.to(device), y.to(device)
                    logits = model(X)
                    val_loss += criterion(logits, y).item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            scheduler.step()
            val_acc = correct / total if total > 0 else 0.0
            avg_train_loss = train_loss / len(train_dl)
            avg_val_loss   = val_loss   / len(val_dl)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                epoch, args.epochs, avg_train_loss, avg_val_loss, val_acc,
            )
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            # Sauvegarde du meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                out_path = Path(phase_cfg["model_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                mlflow.log_artifact(str(out_path))
                logger.info("Meilleur modèle sauvegardé (acc=%.3f)", best_val_acc)

        mlflow.log_metric("best_val_acc", best_val_acc)
        logger.info("Entraînement terminé. Meilleure accuracy : %.3f", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le Phase Classifier")
    parser.add_argument("--data-dir", default="data/annotations", help="Répertoire des clips annotés")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())
