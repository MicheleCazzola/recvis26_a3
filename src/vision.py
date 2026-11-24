import time
from dataclasses import dataclass

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import accuracy, get_logger, safe_tag
from .data import ImageDataset, extract_features, linear_logits, preprocess_input
from .training import make_optimizer, make_scheduler

logger = get_logger()

# Values come from configs/config.yaml (main.load_config); no defaults
@dataclass
class VisionConfig:
    # Model
    library: str  # Either 'torchvision' or 'timm'
    model_name: str
    pretrained: bool
    weights: str  # Only for torchvision; timm takes the weights from model_name
    trainable_blocks: int  # Trainable units: 0 linear probe, k last k blocks, -1 all
    freeze_bn_stats: bool
    image_size: int  # 32 for a ResNet from scratch
    resize_interpolation: str

    # Augmentation (train, 32px)
    augmentations: tuple
    crop: str  # "random_crop" for a ResNet from scratch
    crop_padding: int
    crop_scale: tuple
    hflip_p: float

    # Head
    normalize_features: bool
    head_batch_size: int
    l2sp: float

    # Optimization
    batch_size: int
    epochs: int
    optimizer: str
    lr: float
    backbone_lr: float  # ViT: LR of the last block, LR * layer_decay; ResNet: LR * 0.1
    layer_decay: float
    weight_decay: float
    backbone_weight_decay: float
    backbone_l2sp: float  # Towards the pretrained weights
    momentum: float
    betas: tuple
    scheduler: str  # "cosine", "constant", "onecycle" or None (no scheduler)
    warmup_fraction: float
    label_smoothing: float
    grad_clip: float
    save_best_val_checkpoint: bool
    tta_flip: bool  # Cached features: every split, else validation and test
    features_from: str  # Experiment folder: reuse its cached features
    seed: int

    def __post_init__(self):
        if self.trainable_blocks < -1:
            raise ValueError("The trainable_blocks value must be -1, 0 or a positive number of blocks")
        if not self.pretrained and self.trainable_blocks != -1:
            raise ValueError("Linear probe and partial fine-tuning need pretrained weights (else trainable_blocks=-1)")
        if not self.pretrained and self.backbone_l2sp:
            raise ValueError("The backbone_l2sp option needs pretrained weights")

        self.augmentations = tuple(self.augmentations)

        if {"augmix", "randaugment"} <= set(self.augmentations):
            raise ValueError("Use augmix or randaugment, not both")

        if self.features_from and not self.cached_features:
            raise ValueError("The features_from option needs a linear probe without augmentation")

    @property
    def cached_features(self):
        """Frozen backbone, no augmentation: features are computed once and only the head is trained."""
        return self.trainable_blocks == 0 and not self.augmentations

    @property
    def is_vit(self):
        return "vit" in self.model_name.lower()

    @property
    def name(self):
        mode = {0: "lp", -1: "ft"}.get(self.trainable_blocks, f"ft{self.trainable_blocks}")
        aug = "_aug" if self.augmentations else ""

        return safe_tag(f"{self.model_name}{'_pt' if self.pretrained else ''}_{mode}{aug}")

    def feature_file(self, split):
        return safe_tag("feat", self.name, self.resize_interpolation, f"tta{int(self.tta_flip)}", split)


class BackboneClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, normalize, num_classes):
        super().__init__()
        self.backbone, self.normalize = backbone, normalize
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x):
        features = self.backbone(x)
        return F.normalize(features.float(), dim=-1) if self.normalize else features

    def forward(self, x):
        return self.head(self.features(x))


def build_backbone(cfg, scratch_mean, scratch_std):
    """Backbone without its classification head. Returns (backbone, feature_dim, info with mean/std/weights).

    scratch_mean / scratch_std: input normalization of a randomly initialized model
    """
    if cfg.library == "torchvision":
        weights = models.get_model_weights(cfg.model_name)[cfg.weights] if cfg.pretrained else None
        backbone = models.get_model(cfg.model_name, weights=weights)

        if weights is not None:
            preprocess = weights.transforms()
            native, mean, std = int(preprocess.crop_size[0]), tuple(preprocess.mean), tuple(preprocess.std)
        else:
            native, mean, std = int(getattr(backbone, "image_size", 224)), scratch_mean, scratch_std

        if hasattr(backbone, "fc"):
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "heads"):
            feature_dim = backbone.hidden_dim
            backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Model {cfg.model_name}: only torchvision ResNets and ViTs are supported")

        weights_name = str(weights) if weights is not None else "random init"
    elif cfg.library == "timm":
        options = {"img_size": cfg.image_size} if cfg.image_size and cfg.is_vit else {}
        backbone = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=0, **options)
        pretrained_cfg = backbone.pretrained_cfg
        native = int(cfg.image_size or pretrained_cfg["input_size"][-1])

        if cfg.pretrained:
            mean, std = tuple(pretrained_cfg["mean"]), tuple(pretrained_cfg["std"])
            weights_name = f"timm {pretrained_cfg.get('hf_hub_id') or cfg.model_name}"
        else:
            mean, std, weights_name = scratch_mean, scratch_std, "random init"

        feature_dim = backbone.num_features
    else:
        raise ValueError(f"Library must be 'torchvision' or 'timm', got {cfg.library}")

    if cfg.library == "torchvision" and cfg.is_vit and cfg.image_size != native:
        raise ValueError(f"Model {cfg.model_name} requires image_size={native}")

    modifications = ["classification head removed"]

    if not cfg.pretrained and cfg.image_size == 32 and isinstance(getattr(backbone, "conv1", None), nn.Conv2d):
        backbone.conv1 = nn.Conv2d(3, backbone.conv1.out_channels, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        modifications.append("CIFAR stem: conv1 3x3 stride 1, no maxpool")

    info = {
        "library": cfg.library,
        "architecture": cfg.model_name,
        "class": type(backbone).__name__,
        "weights": weights_name,
        "input_size": cfg.image_size,
        "feature_dim": int(feature_dim),
        "mean": mean,
        "std": std,
        "parameters": sum(p.numel() for p in backbone.parameters()),
        "modifications": modifications,
    }

    return backbone, int(feature_dim), info


def backbone_units(backbone):
    """Units [(name, modules, parameters)] from the input to the output: stem/embedding, block 1, ..., block n (+ final norm)."""
    if hasattr(backbone, "layer4"):  # ResNet (torchvision or timm)
        units = [("stem", [backbone.conv1, backbone.bn1])]
        units += [
            (f"{stage}.{i}", [block])
            for stage in ("layer1", "layer2", "layer3", "layer4")
            for i, block in enumerate(getattr(backbone, stage))
        ]
    elif hasattr(backbone, "blocks"):  # ViT from timm
        units = [
            ("embedding", [m for m in (backbone.patch_embed, getattr(backbone, "norm_pre", None)) if m is not None])
        ]
        units += [(f"blocks.{i}", [block]) for i, block in enumerate(backbone.blocks)]
        units[-1][1].extend(
            m for m in (getattr(backbone, "norm", None), getattr(backbone, "fc_norm", None)) if m is not None
        )
    elif hasattr(backbone, "encoder"):  # ViT from torchvision
        units = [("embedding", [backbone.conv_proj])]
        units += [(f"encoder.layers.{i}", [block]) for i, block in enumerate(backbone.encoder.layers)]
        units[-1][1].append(backbone.encoder.ln)
    else:
        raise ValueError(f"Unknown block structure for {type(backbone).__name__}")

    units = [(name, modules, [p for m in modules for p in m.parameters()]) for name, modules in units]
    covered = {id(p) for _, _, params in units for p in params}

    for name, p in backbone.named_parameters():  # Class token / position embedding -> embedding, the rest -> last unit
        if id(p) not in covered:
            embedding = any(
                key in name for key in ("cls_token", "class_token", "pos_embed", "pos_embedding", "reg_token")
            )
            units[0 if embedding else -1][2].append(p)

    return units


def configure_trainable(model, cfg):
    """Freeze all but the last trainable_blocks units. Returns (optimizer groups, frozen modules, info)."""
    units = backbone_units(model.backbone)
    names = {id(p): name for name, p in model.backbone.named_parameters()}
    k = len(units) if cfg.trainable_blocks == -1 else min(cfg.trainable_blocks, len(units))

    for p in model.backbone.parameters():
        p.requires_grad = False

    groups = [
        {"params": [model.head.weight], "lr": cfg.lr, "weight_decay": cfg.weight_decay, "name": "head"},
        {"params": [model.head.bias], "lr": cfg.lr, "weight_decay": 0.0, "name": "head.bias"},
    ]
    frozen_modules, trainable_units = [], []

    for depth, (name, modules, params) in enumerate(reversed(units), start=1):
        if depth > k:
            frozen_modules += modules
            continue

        for p in params:
            p.requires_grad = True

        lr = cfg.backbone_lr * (cfg.layer_decay ** (depth - 1) if cfg.is_vit and cfg.pretrained else 1.0)
        no_decay = {id(p) for p in params if p.ndim <= 1 or any(t in names[id(p)] for t in ("token", "pos_embed"))}
        decay = [p for p in params if id(p) not in no_decay]

        groups.append(
            {"params": decay, "lr": lr, "weight_decay": cfg.backbone_weight_decay, "l2sp": True, "name": name}
        )
        groups.append(
            {
                "params": [p for p in params if id(p) in no_decay],
                "lr": lr,
                "weight_decay": 0.0,
                "name": f"{name}.no_decay",
            }
        )

        trainable_units.append(name)

    groups = [g for g in groups if g["params"]]
    info = {
        "units": len(units),
        "trainable_units": trainable_units[::-1],
        "lr": {g["name"]: g["lr"] for g in groups if not g["name"].endswith(("bias", "no_decay"))},
        "weight_decay": {"head": cfg.weight_decay, "backbone": cfg.backbone_weight_decay, "biases_norms_tokens": 0.0},
        "l2sp": {"head": cfg.l2sp, "backbone": cfg.backbone_l2sp},
        "trainable_parameters": sum(p.numel() for g in groups for p in g["params"]),
    }
    
    trainable_millions = info["trainable_parameters"] / 1e6
    
    logger.info(f"{cfg.name}: {k}/{len(units)} units trainable, {trainable_millions:.2f}M trainable parameters")

    return groups, frozen_modules, info


def trainable_state_dict(model, groups):
    """State dict restricted to the trainable parameters (of the optimizer groups) and all buffers"""
    trainable = {id(p) for g in groups for p in g["params"]}
    keep = {k for k, p in model.named_parameters() if id(p) in trainable} | {k for k, _ in model.named_buffers()}

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in keep}


# Training and prediction
def predict(model, images, cfg, mean, std, tta_flip, desc, device, amp):
    """Features and logits of a BackboneClassifier, one image at a time (batched)."""
    model.eval()
    features = extract_features(
        model.features,
        images,
        size=cfg.image_size,
        mean=mean,
        std=std,
        interpolation=cfg.resize_interpolation,
        tta_flip=tta_flip,
        normalize=cfg.normalize_features,
        device=device,
        amp=amp,
        desc=desc,
    )

    return features, linear_logits(features, model.head.weight.detach(), model.head.bias.detach(), device=device)


def train_model(
    model,
    groups,
    frozen_modules,
    cfg,
    *,
    mean,
    std,
    transform,
    train_images,
    train_labels,
    val_images,
    val_labels,
    generator,
    worker_init_fn,
    device,
    amp,
    use_grad_scaler,
    num_workers,
):
    """Head + trainable blocks on augmented images; frozen blocks in eval mode. Returns the history."""

    loader = DataLoader(
        ImageDataset(train_images, train_labels, transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    parameters = [p for g in groups for p in g["params"]]
    head_weight_0 = model.head.weight.detach().clone()
    backbone_anchors = [(p, p.detach().clone()) for g in groups if g.get("l2sp") for p in g["params"]]

    opt = make_optimizer(
        groups, cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, betas=cfg.betas
    )
    sched = make_scheduler(opt, cfg.scheduler, cfg.epochs * len(loader), cfg.warmup_fraction)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    best_accuracy, best_state, history = -1.0, None, []

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        if cfg.freeze_bn_stats:
            for module in frozen_modules:
                module.eval()

        start, total_loss, correct, seen = time.time(), 0.0, 0, 0
        progress = tqdm(loader, desc=f"{cfg.name} epoch {epoch}/{cfg.epochs}", unit="batch", dynamic_ncols=True)

        for images, labels in progress:
            inputs = preprocess_input(images, cfg.image_size, mean, std, cfg.resize_interpolation, device)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(**amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)

            # Apply L2SP
            if cfg.l2sp:
                loss = loss + cfg.l2sp * ((model.head.weight - head_weight_0) ** 2).sum()
            if cfg.backbone_l2sp:
                loss = loss + cfg.backbone_l2sp * sum(((p - p0) ** 2).sum() for p, p0 in backbone_anchors)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping (if enabled, by default disabled)
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

            if sched is not None:
                sched.step()

            # Update metrics
            seen += len(labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()

            progress.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

        # Validation inference (logits only)
        _, val_logits = predict(model, val_images, cfg, mean, std, False, "validation", device, amp)
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "train_acc": correct / seen,
            "val_acc": accuracy(val_logits, val_labels),
            "seconds": time.time() - start,
        }
        history.append(row)
        logger.info(f"[{cfg.name}] {row}")

        # Save best validation checkpoint
        if cfg.save_best_val_checkpoint and row["val_acc"] > best_accuracy:
            best_accuracy = row["val_acc"]
            best_state = trainable_state_dict(model, groups)

    # Restore the best validation checkpoint
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        logger.info(f"[{cfg.name}] restored the best CIFAR-10 validation epoch (val_acc={best_accuracy:.4f})")

    return pd.DataFrame(history)
