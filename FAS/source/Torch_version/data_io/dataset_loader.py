"""
Dataset and DataLoader for CASIA-FASD face anti-spoofing.

Filename format: {subject}_{video}.avi_{frame}_{real|fake}.jpg
Label: 'real' → 1, 'fake' → 0

Folder structure:
    train_img/
    └── train_img/
        └── color/
            ├── 10_1.avi_25_real.jpg
            ├── 10_4.avi_25_fake.jpg
            └── ...
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_label_from_filename(filename):
    """Extract label from CASIA-FASD filename.
    
    '10_4.avi_25_fake.jpg' → 0 (fake)
    '10_2.avi_100_real.jpg' → 1 (real)
    """
    stem = Path(filename).stem  # '10_4.avi_25_fake'
    tag = stem.rsplit("_", 1)[-1].lower()  # 'fake' or 'real'
    if tag == "real":
        return 1
    elif tag == "fake":
        return 0
    else:
        return None


def find_color_dir(base_dir):
    """Find color/ directory handling nested structure.
    e.g. train_img/train_img/color/
    """
    base = Path(base_dir)
    # Direct
    if (base / "color").is_dir():
        return base / "color"
    # One level nested
    for sub in base.iterdir():
        if sub.is_dir():
            if (sub / "color").is_dir():
                return sub / "color"
    return None


class FASDataset(Dataset):
    """CASIA-FASD Face Anti-Spoofing dataset.

    Args:
        root_dir: path containing train_img/ or test_img/ structure
        img_size: (H, W) resize target
        ft_size: (H, W) for Fourier map. None = disabled.
        transform: custom transform
        is_train: enables augmentation
    """

    def __init__(self, root_dir, img_size=(80, 80), ft_size=None,
                 transform=None, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.ft_size = ft_size
        self.is_train = is_train

        if transform is not None:
            self.transform = transform
        elif is_train:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        # For FT computation (un-normalized)
        self.raw_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        # Collect samples
        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        root = Path(self.root_dir)
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}

        # Find color directory
        color_dir = find_color_dir(root)
        if color_dir is None:
            # Maybe root itself contains images directly
            color_dir = root

        if not color_dir.exists():
            raise FileNotFoundError(f"Color directory not found: {root}")

        skipped = 0
        for f in sorted(color_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            label = get_label_from_filename(f.name)
            if label is None:
                skipped += 1
                continue
            self.samples.append((str(f), label))

        n_real = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"[Dataset] {self.root_dir}")
        print(f"  Color dir: {color_dir}")
        print(f"  Total: {len(self.samples)} (real={n_real}, fake={n_fake}, skipped={skipped})")

    def __len__(self):
        return len(self.samples)

    def _compute_ft_map(self, img_tensor):
        """Compute Fourier Transform magnitude spectrum as supervision target."""
        # RGB → grayscale
        gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        gray_np = gray.numpy()

        # 2D FFT → magnitude spectrum
        f_shift = np.fft.fftshift(np.fft.fft2(gray_np))
        magnitude = np.log1p(np.abs(f_shift))

        # Normalize to [0, 1]
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)

        # Resize to ft_size
        mag_pil = Image.fromarray((magnitude * 255).astype(np.uint8))
        mag_pil = mag_pil.resize((self.ft_size[1], self.ft_size[0]), Image.BILINEAR)
        ft_map = np.array(mag_pil, dtype=np.float32) / 255.0

        return torch.tensor(ft_map).unsqueeze(0)  # (1, ft_h, ft_w)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        img_tensor = self.transform(img)

        if self.ft_size is not None:
            raw_tensor = self.raw_transform(img)
            ft_map = self._compute_ft_map(raw_tensor)
            return img_tensor, ft_map, label

        return img_tensor, label


def get_train_loader(train_root, img_size=(80, 80), ft_size=None,
                     batch_size=128, num_workers=4):
    dataset = FASDataset(train_root, img_size, ft_size, is_train=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, num_workers=num_workers, drop_last=True)


def get_test_loader(test_root, img_size=(80, 80), batch_size=128, num_workers=4):
    dataset = FASDataset(test_root, img_size, ft_size=None, is_train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      pin_memory=True, num_workers=num_workers)