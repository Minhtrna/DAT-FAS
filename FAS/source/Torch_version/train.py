"""
Training script for Face Anti-Spoofing with MobileNeXt backbone.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 64 --lr 0.01
    python train.py --train_root /path/to/train_img --test_root /path/to/test_img
"""

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from default_config import get_default_config, make_dirs
from data_io.dataset_loader import get_train_loader, get_test_loader
from Model.MultiFTNet import MultiFTNet


class SpectralLoss(nn.Module):
    """Frequency-domain loss comparing magnitude and phase of FFT."""
    def __init__(self, magnitude_weight=1.0, phase_weight=0.5):
        super().__init__()
        self.mag_w = magnitude_weight
        self.phase_w = phase_weight

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        return self.mag_w * F.mse_loss(pred_mag, target_mag) + self.phase_w * F.mse_loss(pred_phase, target_phase)


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss — adaptively weights hard-to-predict frequencies."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        freq_distance = torch.abs(pred_fft - target_fft)
        weight = freq_distance.detach()
        weight = weight / (weight.mean(dim=(-2, -1), keepdim=True) + 1e-8)
        weight = weight ** self.alpha
        return (weight * freq_distance).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train FAS Model")
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument("--test_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--no_ft", action="store_true", help="Disable FT supervision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, test_loader, device):
    """Evaluate model. Returns accuracy, APCER, BPCER, ACER."""
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[-1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for pred, gt in zip(predicted, labels):
                if gt == 1 and pred == 1:
                    tp += 1
                elif gt == 0 and pred == 1:
                    fp += 1
                elif gt == 0 and pred == 0:
                    tn += 1
                elif gt == 1 and pred == 0:
                    fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    acer = (apcer + bpcer) / 2.0

    model.train()
    return acc, apcer, bpcer, acer


def train():
    args = parse_args()
    seed_everything(args.seed)
    conf = get_default_config()

    # Override config with CLI args
    if args.train_root:
        conf["train_root"] = args.train_root
    if args.test_root:
        conf["test_root"] = args.test_root
    if args.epochs:
        conf["epochs"] = args.epochs
    if args.batch_size:
        conf["batch_size"] = args.batch_size
    if args.lr:
        conf["lr"] = args.lr
    if args.no_ft:
        conf["use_ft"] = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(conf["device"])
    make_dirs(conf)

    # Print config
    print("=" * 60)
    print("Configuration:")
    for k, v in conf.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Data
    ft_size = tuple(conf["ft_size"]) if conf["use_ft"] else None
    train_loader = get_train_loader(
        train_root=conf["train_root"],
        img_size=tuple(conf["img_size"]),
        ft_size=ft_size,
        batch_size=conf["batch_size"],
        num_workers=conf["num_workers"],
    )
    test_loader = get_test_loader(
        test_root=conf["test_root"],
        img_size=tuple(conf["img_size"]),
        batch_size=conf["batch_size"],
        num_workers=conf["num_workers"],
    )

    # Model
    model = MultiFTNet(
        num_classes=conf["num_classes"],
        img_channel=conf["img_channel"],
        embedding_size=conf["embedding_size"],
        ft_size=ft_size,
    )
    model = model.to(device)
    if len(conf["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=conf["gpu_ids"])

    # Loss & Optimizer
    cls_criterion = nn.CrossEntropyLoss()
    ft_criterion = nn.MSELoss()
    spectral_criterion = SpectralLoss()
    focal_freq_criterion = FocalFrequencyLoss(alpha=1.0)

    # CDC-optimized: Adam + CosineAnnealing
    optimizer = optim.Adam(
        model.parameters(),
        lr=conf["lr"],
        weight_decay=5e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=conf["epochs"], eta_min=1e-6
    )

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded weights from {args.resume}")

    # Training loop
    best_acer = 1.0
    for epoch in range(start_epoch, conf["epochs"]):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_ft_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{conf['epochs']}")
        for batch_idx, batch_data in enumerate(pbar):
            if conf["use_ft"]:
                images, ft_targets, labels = batch_data
                ft_targets = ft_targets.to(device)
            else:
                images, labels = batch_data

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if conf["use_ft"]:
                cls_out, ft_out = model(images)
                loss_cls = cls_criterion(cls_out, labels)
                loss_mse = ft_criterion(ft_out, ft_targets)
                loss_spectral = spectral_criterion(ft_out, ft_targets)
                loss_focal = focal_freq_criterion(ft_out, ft_targets)
                # Scaled FT losses: total FT weight = 0.5 (0.3 + 0.1 + 0.1)
                loss_ft = loss_mse
                loss = (conf["cls_weight"] * loss_cls
                        + 0.3 * loss_mse
                        + 0.1 * loss_spectral
                        + 0.1 * loss_focal)
            else:
                cls_out = model(images)
                loss_cls = cls_criterion(cls_out, labels)
                loss_ft = torch.tensor(0.0)
                loss = loss_cls

            loss.backward()
            optimizer.step()

            # Statistics
            _, predicted = torch.max(cls_out, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            running_cls_loss += loss_cls.item()
            running_ft_loss += loss_ft.item()

            pbar.set_postfix({
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc": f"{running_correct / running_total:.4f}",
            })

        scheduler.step()

        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_correct / running_total
        epoch_time = time.time() - epoch_start

        print(f"\n[Epoch {epoch + 1}] "
              f"Loss: {epoch_loss:.4f} | "
              f"CLS: {running_cls_loss / len(train_loader):.4f} | "
              f"FT: {running_ft_loss / len(train_loader):.4f} | "
              f"Acc: {epoch_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Evaluate
        acc, apcer, bpcer, acer = evaluate(model, test_loader, device)
        print(f"[Test]  Acc: {acc:.4f} | APCER: {apcer:.4f} | "
              f"BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

        # Save best
        if acer < best_acer:
            best_acer = acer
            save_path = os.path.join(conf["save_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acer": acer,
                "acc": acc,
            }, save_path)
            print(f"  ★ Best model saved (ACER: {acer:.4f})")

        # Periodic save
        if (epoch + 1) % conf["save_interval"] == 0:
            save_path = os.path.join(conf["save_dir"], f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

    print(f"\nTraining complete. Best ACER: {best_acer:.4f}")


if __name__ == "__main__":
    train()