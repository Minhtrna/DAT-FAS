"""
Quick sanity check before training.
Verifies: dataset loading, model forward pass, loss computation.

Usage:
    python verify_setup.py
"""

import torch
from default_config import get_default_config
from data_io.dataset_loader import get_train_loader, get_test_loader
from Model.MultiFTNet import MultiFTNet


def main():
    conf = get_default_config()
    device = torch.device(conf["device"])
    ft_size = tuple(conf["ft_size"]) if conf["use_ft"] else None

    # 1. Check dataset
    print("=" * 60)
    print("1. Loading datasets...")
    print("=" * 60)
    train_loader = get_train_loader(
        conf["train_root"], tuple(conf["img_size"]), ft_size,
        batch_size=4, num_workers=0,
    )
    test_loader = get_test_loader(
        conf["test_root"], tuple(conf["img_size"]),
        batch_size=4, num_workers=0,
    )

    # 2. Check a batch
    print("\n" + "=" * 60)
    print("2. Checking train batch...")
    print("=" * 60)
    batch = next(iter(train_loader))
    if conf["use_ft"]:
        images, ft_maps, labels = batch
        print(f"  Images:  {images.shape}")   # [4, 3, 80, 80]
        print(f"  FT maps: {ft_maps.shape}")  # [4, 1, 10, 10]
        print(f"  Labels:  {labels}")          # tensor([1, 0, 1, 0])
    else:
        images, labels = batch
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels}")

    print(f"\n  Test batch:")
    test_batch = next(iter(test_loader))
    test_images, test_labels = test_batch
    print(f"  Images: {test_images.shape}")
    print(f"  Labels: {test_labels}")

    # 3. Check model
    print("\n" + "=" * 60)
    print("3. Checking model...")
    print("=" * 60)
    model = MultiFTNet(
        num_classes=conf["num_classes"],
        img_channel=conf["img_channel"],
        embedding_size=conf["embedding_size"],
        ft_size=ft_size,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {train_params:,}")

    # Check backbone split
    x = torch.randn(1, 3, 80, 80).to(device)
    early_out = model.early_features(x)
    late_out = model.late_features(early_out)
    print(f"\n  Input:          {x.shape}")
    print(f"  Early features: {early_out.shape}")  # should be [1, 128, H, W]
    print(f"  Late features:  {late_out.shape}")   # should be [1, 512, H, W]

    # 4. Forward pass
    print("\n" + "=" * 60)
    print("4. Forward pass (train mode)...")
    print("=" * 60)
    model.train()
    images = images.to(device)
    cls_out, ft_out = model(images)
    print(f"  Classification: {cls_out.shape}")  # [4, 2]
    print(f"  FT output:      {ft_out.shape}")   # [4, 1, 10, 10]

    # 5. Loss computation
    print("\n" + "=" * 60)
    print("5. Loss computation...")
    print("=" * 60)
    cls_criterion = torch.nn.CrossEntropyLoss()
    ft_criterion = torch.nn.MSELoss()

    labels = labels.to(device)
    loss_cls = cls_criterion(cls_out, labels)
    if conf["use_ft"]:
        ft_maps = ft_maps.to(device)
        loss_ft = ft_criterion(ft_out, ft_maps)
        loss = conf["cls_weight"] * loss_cls + conf["ft_weight"] * loss_ft
        print(f"  CLS loss: {loss_cls.item():.4f}")
        print(f"  FT loss:  {loss_ft.item():.4f}")
        print(f"  Total:    {loss.item():.4f}")
    else:
        print(f"  CLS loss: {loss_cls.item():.4f}")

    # 6. Backward
    loss.backward()
    print("  Backward: OK")

    # 7. Eval mode
    print("\n" + "=" * 60)
    print("6. Forward pass (eval mode)...")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        cls_only = model(images)
    print(f"  Output: {cls_only.shape}")  # [4, 2]
    probs = torch.softmax(cls_only, dim=1)
    print(f"  Probs:  {probs}")

    print("\n" + "=" * 60)
    print("✅ All checks passed! Ready to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()