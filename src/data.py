from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import get_logger

logger = get_logger()


# ---- Dataset and augmentations ----

AUGMENTATIONS = ("crop", "flip", "color", "grayscale", "augmix", "randaugment")


def load_npz(path):
    """Returns (images, labels, ids)"""
    with np.load(path) as archive:
        images = archive["images"].astype(np.uint8)
        labels = archive["labels"].astype(np.int64) if "labels" in archive.files else None
        ids = archive["ids"].astype(np.int64) if "ids" in archive.files else np.arange(len(images))

    return images, labels, ids


class ImageDataset(Dataset):

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        return torch.from_numpy(np.array(image, dtype=np.uint8)), int(self.labels[index])


def build_augmentation(cfg, available=AUGMENTATIONS):
    """Training transform at 32px: the augmentations of cfg"""
    unknown = set(cfg.augmentations) - set(available)

    if unknown:
        raise ValueError(f"Unknown augmentations {sorted(unknown)}; choose from {available}")

    ops = {
        "crop": (
            T.RandomCrop(32, padding=cfg.crop_padding)
            if cfg.crop == "random_crop"
            else T.RandomResizedCrop(32, scale=tuple(cfg.crop_scale))
        ),
        "flip": T.RandomHorizontalFlip(p=cfg.hflip_p),
        "color": T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        "grayscale": T.RandomGrayscale(p=0.2),
        "augmix": T.AugMix(severity=3, mixture_width=3),
        "randaugment": T.RandAugment(num_ops=2, magnitude=9),
    }

    return T.Compose([ops[name] for name in available if name in cfg.augmentations])


# ---- Features ----

def preprocess_input(batch, size, mean, std, interpolation, device):
    x = batch.to(device, non_blocking=True)

    if x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)

    x = x.float().div_(255)

    if x.shape[-1] != size:
        align = False if interpolation in ("bilinear", "bicubic") else None
        x = F.interpolate(x, size=(size, size), mode=interpolation, align_corners=align).clamp_(0, 1)

    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    return (x - mean) / std


@torch.no_grad()
def extract_features(
    feature_fn, images, *, size, mean, std, interpolation, tta_flip, normalize, device, amp, batch_size=256, desc="features"
):
    """Features of images. With tta_flip, mean of the image and its horizontal flip."""
    images = torch.from_numpy(images)
    outputs = []

    for first in tqdm(range(0, len(images), batch_size), desc=desc, unit="batch", dynamic_ncols=True):
        batch = images[first : first + batch_size]
        views = []

        for view in ([batch, batch.flip(2)] if tta_flip else [batch]):
            with torch.autocast(**amp):
                features = feature_fn(preprocess_input(view, size, mean, std, interpolation, device)).float()

            views.append(F.normalize(features, dim=-1) if normalize else features)

        features = torch.stack(views).mean(0)
        outputs.append((F.normalize(features, dim=-1) if normalize else features).cpu())

    return torch.cat(outputs)


@torch.no_grad()
def linear_logits(features, weight, bias=None, *, device, batch_size=8192):
    weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
    features = torch.as_tensor(features, dtype=torch.float32)
    
    if bias is not None:
        bias = torch.as_tensor(bias, dtype=torch.float32, device=device)
    
    outputs = []
    for chunk in features.split(batch_size):
        logits = chunk.to(device, dtype=torch.float32) @ weight.T
        outputs.append((logits if bias is None else logits + bias).cpu())

    return torch.cat(outputs)


def load_features(folder, name, rows):
    """Features <folder>/cache/<name>.npy of a previous experiment, or None (folder None or file missing)."""
    
    if folder is None:
        return None

    path = Path(folder) / "cache" / f"{name}.npy"

    if not path.is_file():
        logger.warning(f"Features {path} not found: computing them")
        return None

    features = torch.from_numpy(np.load(path).astype(np.float32))

    if len(features) != rows:
        raise ValueError(f"File {path} has {len(features)} rows, expected {rows}")

    logger.info(f"loaded {path} {tuple(features.shape)}")

    return features
