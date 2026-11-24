import dataclasses
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import timm
import torch
import torchvision
import yaml

# ----- Configuration and device -----
DEFAULT_SEED = 42
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")        # CIFAR-10 classes


@dataclass
class GlobalConfig:
    input_dir: Path  # train.npz, validation.npz, test.npz (configs/config.yaml input_dir, CLI --input-dir)
    output_dir: Path  # Run folders (configs/config.yaml output_dir, CLI --output-dir)
    seed: int = DEFAULT_SEED
    class_names: tuple = CLASS_NAMES
    cifar_mean: tuple = (0.4914, 0.4822, 0.4465)
    cifar_std: tuple = (0.2470, 0.2435, 0.2616)
    num_workers: int = min(4, os.cpu_count() or 2)

    def __post_init__(self):
        self.input_dir, self.output_dir = Path(self.input_dir), Path(self.output_dir)

    @property
    def num_classes(self):
        return len(self.class_names)


def setup_device(device=None):
    """Returns (device, amp, use_grad_scaler).

    amp: kwargs of torch.autocast (bfloat16 on Ampere+ GPUs, float16 on older ones, disabled on CPU)
    use_grad_scaler: True only for float16 autocast
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    amp = dict(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")
    use_grad_scaler = amp["enabled"] and dtype == torch.float16

    return device, amp, use_grad_scaler


# --- Utilities -----

LOGGER_NAME = "recvis"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(level=logging.INFO):
    """Configures the root logger (console) and returns the 'recvis' logger."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT, force=True)
    return logging.getLogger(LOGGER_NAME)


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator, seed_worker


def safe_tag(*parts):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(str(p) for p in parts))


def to_plain(obj):
    """Configs and results -> YAML-friendly Python objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return np.asarray(obj).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, str):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    return str(obj)


def environment_info(device, amp):
    info = {
        "device": str(device),
        "autocast": str(amp["dtype"]) if amp["enabled"] else "float32",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "open_clip": open_clip.__version__,
    }

    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)

    return info

def accuracy(logits, labels):
    return float((torch.as_tensor(logits).argmax(1) == torch.as_tensor(labels)).float().mean())


logger = get_logger()


# ----- Run -----
class Run:
    """Folder results/<YYYYmmdd_HHMM>/ with config.yaml, run.log, submission_*.csv, logits/, checkpoints/, cache/, *.csv"""

    def __init__(self, section, config, global_config, environment):
        self.section, self.config = section, config
        self.global_config, self.environment = global_config, environment
        self.started = datetime.now()
        self.dir = self.new_folder()

        for sub in ("logits", "checkpoints", "cache"):
            (self.dir / sub).mkdir()

        self.results, self.files, self.metrics = {}, [], []
        self.attach_log_file()
        self.generator, self.worker_init_fn = set_seed(config.seed)
        self.save_config()
        logger.info(f"run folder {self.dir} | {to_plain(config)}")

    def attach_log_file(self):
        """Replaces any previous run.log handler of the root logger with this run's one."""
        root = logging.getLogger()

        for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
            root.removeHandler(handler)
            handler.close()

        console = [h for h in root.handlers if h.formatter is not None]
        formatter = console[0].formatter if console else logging.Formatter(LOG_FORMAT, LOG_DATEFMT)
        self.log_handler = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        self.log_handler.setFormatter(formatter)
        root.addHandler(self.log_handler)

    def new_folder(self):
        output_dir = self.global_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started.strftime("%Y%m%d_%H%M")

        for name in [stamp] + [f"{self.started.strftime('%Y%m%d_%H%M%S')}_{i}" for i in range(1, 100)]:
            try:
                (output_dir / name).mkdir()
                return output_dir / name
            except FileExistsError:
                continue

        raise RuntimeError(f"No free run folder in {output_dir}")

    def path(self, *parts):
        path = self.dir.joinpath(*parts)
        relative = path.relative_to(self.dir).as_posix()

        if relative not in self.files:
            self.files.append(relative)

        return path

    def save_config(self):
        content = to_plain(
            {
                "section": self.section,
                "started": self.started.isoformat(timespec="seconds"),
                "environment": self.environment,
                "global_config": self.global_config,
                "section_config": self.config,
                "results": self.results,
                "files": self.files,
            }
        )
        text = yaml.safe_dump(content, sort_keys=False, allow_unicode=True, width=120)
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            f.write(text)

    def record(self, key, value):
        self.results[key] = to_plain(value)
        self.save_config()

    def save_features(self, name, features):
        np.save(self.path("cache", f"{name}.npy"), torch.as_tensor(features).numpy().astype(np.float16))

    def save_submission(self, name, test_logits, val_logits, val_labels, test_ids):
        """Writes submission_<name>.csv (columns id, label; row i = test image i), logits/<name>.npy (test),
        logits/<name>_validation.npy and a row of metrics.csv"""
        test_logits, val_logits = torch.as_tensor(test_logits).float(), torch.as_tensor(val_logits).float()
        predictions = test_logits.argmax(1).numpy()

        if len(predictions) != len(test_ids):
            raise ValueError(f"Got {len(predictions)} predictions for {len(test_ids)} test images")

        submission = pd.DataFrame({"id": np.asarray(test_ids, dtype=np.int64), "label": predictions.astype(np.int64)})
        submission.to_csv(self.path(f"submission_{name}.csv"), index=False)
        np.save(self.path("logits", f"{name}.npy"), test_logits.numpy())
        np.save(self.path("logits", f"{name}_validation.npy"), val_logits.numpy())

        metrics = {
            "name": name,
            "cifar_val_acc": accuracy(val_logits, val_labels),
            "val_mean_max_prob": round(val_logits.softmax(1).max(1).values.mean().item(), 4),
            "test_mean_max_prob": round(test_logits.softmax(1).max(1).values.mean().item(), 4),
            "test_class_shares": (np.bincount(predictions, minlength=test_logits.shape[1]) / len(predictions)).round(4),
        }
        self.metrics.append(metrics)
        self.save_table("metrics", [{k: v for k, v in m.items() if k != "test_class_shares"} for m in self.metrics])
        self.record(f"submission_{name}", metrics)
        logger.info(f"saved submission_{name}.csv | {metrics}")

        return metrics

    def save_table(self, name, table):
        pd.DataFrame(table).to_csv(self.path(f"{name}.csv"), index=False)
        self.save_config()

    def save_checkpoint(self, name, obj):
        torch.save(obj, self.path("checkpoints", f"{name}.pth"))
        self.save_config()

    def finish(self):
        self.record("duration_seconds", round((datetime.now() - self.started).total_seconds(), 1))
        logger.info(f"run finished: {self.dir} ({len(self.files)} files)")
        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler.close()
