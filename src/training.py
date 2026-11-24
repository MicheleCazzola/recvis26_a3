import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .utils import accuracy, get_logger
from .data import linear_logits

logger = get_logger()


# Optimizers
def make_optimizer(groups, name, *, lr, weight_decay, momentum=0.9, betas=(0.9, 0.999)):
    """Groups: list of {'params', 'lr', 'name'} (one LR per group)."""
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=tuple(betas))
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay, betas=tuple(betas))
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, weight_decay=weight_decay, momentum=momentum)

    raise ValueError(f"Unknown optimizer {name}")


# Schedulers
def make_scheduler(optimizer, name, total_steps, warmup_fraction=0.0):
    """Stepped per batch; linear warmup over warmup_fraction of the steps, then cosine or constant."""

    # No scheduler
    if name is None:
        return None

    total_steps = max(1, total_steps)

    if name == "onecycle":
        if total_steps < 3:
            raise ValueError("The onecycle scheduler needs at least 3 steps")

        max_lr = [group["lr"] for group in optimizer.param_groups]
        pct_start = min(max(warmup_fraction, 2 / total_steps), (total_steps - 1) / total_steps)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start
        )

    if name not in ("cosine", "constant"):
        raise ValueError(f"Unknown scheduler {name}")

    warmup_steps = round(warmup_fraction * total_steps)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "constant":
            return 1.0

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# Training head
def train_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    *,
    epochs,
    batch_size,
    optimizer,
    lr,
    weight_decay,
    scheduler,
    num_classes,
    device,
    weight_init=None,
    momentum=0.9,
    betas=(0.9, 0.999),
    warmup_fraction=0.0,
    label_smoothing=0.0,
    l2sp=0.0,
    grad_clip=None,
    save_best_val_checkpoint=False,
    generator=None,
    desc="linear head",
):
    train_features = torch.as_tensor(train_features, dtype=torch.float32).to(device)
    train_labels = torch.as_tensor(train_labels, dtype=torch.long).to(device)
    head = nn.Linear(train_features.shape[1], num_classes).to(device)

    if weight_init is not None:
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(weight_init, dtype=torch.float32))
            head.bias.zero_()

    weight_0 = head.weight.detach().clone()
    steps_per_epoch = math.ceil(len(train_features) / batch_size)

    opt = make_optimizer(
        [
            {"params": [head.weight], "lr": lr, "weight_decay": weight_decay, "name": "head"},
            {"params": [head.bias], "lr": lr, "weight_decay": 0.0, "name": "head.bias"},
        ],
        optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        betas=betas,
    )
    sched = make_scheduler(opt, scheduler, epochs * steps_per_epoch, warmup_fraction)
    best_accuracy, best_state, history = -1.0, None, []

    for epoch in tqdm(range(1, epochs + 1), desc=desc, unit="epoch", dynamic_ncols=True):
        total_loss, correct = 0.0, 0

        # Simulate shuffling by random permutation of the indices (to avoid copying the features)
        for index in torch.randperm(len(train_features), generator=generator).to(device).split(batch_size):
            logits = head(train_features[index])
            loss = F.cross_entropy(logits, train_labels[index], label_smoothing=label_smoothing)

            # L2-SP regularization
            if l2sp:
                loss = loss + l2sp * ((head.weight - weight_0) ** 2).sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)

            opt.step()
            if sched is not None:
                sched.step()

            # Update metrics
            total_loss += loss.item() * len(index)
            correct += (logits.argmax(1) == train_labels[index]).sum().item()

        # Validation inference
        val_logits = linear_logits(val_features, head.weight.detach(), head.bias.detach(), device=device)
        val_acc = accuracy(val_logits, val_labels)

        # Compute drift from the initial head and save metrics
        drift = float((head.weight.detach() - weight_0).norm() / weight_0.norm().clamp_min(1e-8))
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(train_features),
                "train_acc": correct / len(train_features),
                "val_acc": val_acc,
                "drift": drift,
            }
        )

        # Save the best validation checkpoint
        if save_best_val_checkpoint and val_acc > best_accuracy:
            best_accuracy, best_state = val_acc, {k: v.detach().clone() for k, v in head.state_dict().items()}

    logger.info(f"[{desc}] {history[-1]}")

    if best_state is not None:
        head.load_state_dict(best_state)
        logger.info(f"[{desc}] restored the best CIFAR-10 validation epoch (val_acc={best_accuracy:.4f})")

    return head.weight.detach().cpu(), head.bias.detach().cpu(), pd.DataFrame(history)


def wise_interpolate(weight_0, weight_1, bias_1, alpha):
    """WiSE-FT: W = (1 - a) W_0 + a W_1, b = a b_1 (the zero-shot head has no bias)."""
    return (1 - alpha) * weight_0 + alpha * weight_1, alpha * bias_1
