import argparse
import dataclasses
from pathlib import Path

import torch
import yaml

from src.vision import (
    BackboneClassifier,
    VisionConfig,
    build_backbone,
    configure_trainable,
    predict,
    train_model,
    trainable_state_dict,
)
from src.clip import PROMPT_SETS, CLIPConfig, load_clip, text_classifier
from src.utils import (
    GlobalConfig,
    Run,
    accuracy,
    environment_info,
    get_logger,
    setup_device,
    setup_logging,
    to_plain,
)
from src.data import (
    AUGMENTATIONS,
    build_augmentation,
    extract_features,
    linear_logits,
    load_features,
    load_npz,
)
from src.training import train_head, wise_interpolate

logger = get_logger()

DEFAULT_CONFIG = Path("./configs/config.yaml")  # Relative to the working directory, like every path
PIPELINES = {"vision": VisionConfig, "clip": CLIPConfig}
DIR_KEYS = ("input_dir", "output_dir")  # GlobalConfig attributes of the config file


def load_data(cfg):
    """Train / validation / test images and labels, and the test ids (id column of the submissions)"""
    train_images, train_labels, _ = load_npz(cfg.input_dir / "train.npz")
    val_images, val_labels, _ = load_npz(cfg.input_dir / "validation.npz")
    test_images, test_labels, test_ids = load_npz(cfg.input_dir / "test.npz")

    if test_labels is not None:
        raise ValueError("test.npz must not contain labels: test labels must never be used")

    data = {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_ids": test_ids,
    }
    logger.info(f"train {train_images.shape} | validation {val_images.shape} | test {test_images.shape}")

    return data


def run_vision(global_cfg, cfg, data, device, amp, use_grad_scaler):
    """Vision pipeline: ResNet / ViT.

    A frozen backbone without augmentation caches the features and trains only the head.
    """

    run = Run("vision", cfg, global_cfg, environment_info(device, amp))

    backbone, feature_dim, info = build_backbone(cfg, global_cfg.cifar_mean, global_cfg.cifar_std)
    model = BackboneClassifier(backbone, feature_dim, cfg.normalize_features, global_cfg.num_classes).to(device)

    groups, frozen_modules, trainable = configure_trainable(model, cfg)
    mean, std = info["mean"], info["std"]

    run.record("model", {**info, "trainable": trainable})

    # Training and validation
    if cfg.cached_features:
        model.eval()
        cached = {}

        for split, images in (("train", data["train_images"]), ("validation", data["val_images"])):
            features = load_features(cfg.features_from, cfg.feature_file(split), len(images))

            # If features are not available, extract them with the backbone
            if features is None:
                features = extract_features(
                    model.features,
                    images,
                    size=cfg.image_size,
                    mean=mean,
                    std=std,
                    interpolation=cfg.resize_interpolation,
                    tta_flip=cfg.tta_flip,
                    normalize=cfg.normalize_features,
                    device=device,
                    amp=amp,
                    desc=f"{split} features",
                )
                run.save_features(cfg.feature_file(split), features)

            cached[split] = features

        # Train head (linear probe)
        weight, bias, history = train_head(
            cached["train"],
            data["train_labels"],
            cached["validation"],
            data["val_labels"],
            epochs=cfg.epochs,
            batch_size=cfg.head_batch_size,
            optimizer=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            scheduler=cfg.scheduler,
            num_classes=global_cfg.num_classes,
            device=device,
            momentum=cfg.momentum,
            betas=cfg.betas,
            warmup_fraction=cfg.warmup_fraction,
            label_smoothing=cfg.label_smoothing,
            l2sp=cfg.l2sp,
            grad_clip=cfg.grad_clip,
            save_best_val_checkpoint=cfg.save_best_val_checkpoint,
            generator=run.generator,
            desc=cfg.name,
        )

        with torch.no_grad():
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)

        # Validation inference
        val_logits = linear_logits(cached["validation"], weight, bias, device=device)
    else:

        # Train (fine-tune) the whole model, with or without augmentation
        history = train_model(
            model,
            groups,
            frozen_modules,
            cfg,
            mean=mean,
            std=std,
            transform=build_augmentation(cfg, AUGMENTATIONS),
            train_images=data["train_images"],
            train_labels=data["train_labels"],
            val_images=data["val_images"],
            val_labels=data["val_labels"],
            generator=run.generator,
            worker_init_fn=run.worker_init_fn,
            device=device,
            amp=amp,
            use_grad_scaler=use_grad_scaler,
            num_workers=global_cfg.num_workers,
        )

        # Validation inference
        val_features, val_logits = predict(
            model, data["val_images"], cfg, mean, std, cfg.tta_flip, "validation", device, amp
        )
        run.save_features(cfg.feature_file("validation"), val_features)

    # Save results and checkpoint
    run.save_table("history", history)
    run.record("cifar_val_acc", accuracy(val_logits, data["val_labels"]))
    run.save_checkpoint(
        cfg.name,
        {
            "config": to_plain(cfg),
            "model": info,
            "trainable": trainable,
            "state_dict": trainable_state_dict(model, groups),
        },
    )

    # Test
    test_features = load_features(cfg.features_from, cfg.feature_file("test"), len(data["test_images"]))

    if test_features is None:
        test_features, test_logits = predict(
            model, data["test_images"], cfg, mean, std, cfg.tta_flip, "test", device, amp
        )
        run.save_features(cfg.feature_file("test"), test_features)
    else:
        test_logits = linear_logits(test_features, model.head.weight.detach(), model.head.bias.detach(), device=device)

    run.save_submission(cfg.name, test_logits, val_logits, data["val_labels"], data["test_ids"])
    run.finish()

    return run


def run_clip(global_cfg, cfg, data, device, amp):
    """CLIP zero-shot; with wise_alpha, also a linear probe initialized from it and WiSE-FT."""

    templates = PROMPT_SETS[cfg.prompt_set]
    run = Run("clip", cfg, global_cfg, environment_info(device, amp))

    model, tokenizer, info = load_clip(cfg, device)
    text = text_classifier(model, tokenizer, templates, global_cfg.class_names, device, amp)

    run.record("model", info)
    run.record("prompts", {"set": cfg.prompt_set, "templates": templates})

    splits = {"validation": data["val_images"], "test": data["test_images"]}

    if cfg.wise_alpha is not None:
        splits["train"] = data["train_images"]

    features = {}

    for split, images in splits.items():
        cached = load_features(cfg.features_from, cfg.feature_file(split), len(images))

        if cached is None:
            cached = extract_features(
                model.encode_image,
                images,
                size=info["input_size"],
                mean=info["mean"],
                std=info["std"],
                interpolation=cfg.interpolation,
                tta_flip=cfg.tta_flip,
                normalize=True,
                device=device,
                amp=amp,
                batch_size=cfg.batch_size,
                desc=f"CLIP {split}",
            )
            run.save_features(cfg.feature_file(split), cached)

        features[split] = cached

    # Zero-shot
    w0 = info["logit_scale"] * text
    zeroshot_val = linear_logits(features["validation"], w0, device=device)
    zeroshot_test = linear_logits(features["test"], w0, device=device)
    run.save_submission(cfg.zeroshot_name, zeroshot_test, zeroshot_val, data["val_labels"], data["test_ids"])

    # Linear probe + WiSE-FT
    if cfg.wise_alpha is not None:
        w1, b1, history = train_head(
            features["train"],
            data["train_labels"],
            features["validation"],
            data["val_labels"],
            weight_init=w0,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size_head,
            optimizer=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            scheduler=cfg.scheduler,
            num_classes=global_cfg.num_classes,
            device=device,
            warmup_fraction=cfg.warmup_fraction,
            label_smoothing=cfg.label_smoothing,
            l2sp=cfg.l2sp,
            save_best_val_checkpoint=cfg.save_best_val_checkpoint,
            generator=run.generator,
            desc="CLIP head",
        )
        run.save_table("head_history", history)
        run.save_checkpoint(
            f"clip_heads_{cfg.tag}_prompt_{cfg.prompt_tag}",
            {
                "zeroshot_weight": w0,
                "finetuned_weight": w1,
                "finetuned_bias": b1,
                "config": to_plain(cfg),
                "model": info,
            },
        )

        # WiSE-FT interpolation
        weight, bias = wise_interpolate(w0, w1, b1, cfg.wise_alpha)

        # Inference
        val_logits = linear_logits(features["validation"], weight, bias, device=device)
        test_logits = linear_logits(features["test"], weight, bias, device=device)

        # Save submission and record metrics
        run.save_submission(cfg.name, test_logits, val_logits, data["val_labels"], data["test_ids"])
        run.record(
            "wise",
            {
                "alpha": cfg.wise_alpha,
                "cifar_val_acc": accuracy(val_logits, data["val_labels"]),
                "test_agreement_with_zeroshot": float(
                    (test_logits.argmax(1) == zeroshot_test.argmax(1)).float().mean()
                ),
                "test_mean_confidence": test_logits.softmax(1).max(1).values.mean().item(),
            },
        )

    run.finish()

    return run


def load_config(path, pipeline=None, model=None, pretrained=None, input_dir=None, output_dir=None):
    """Pipeline, VisionConfig or CLIPConfig, and GlobalConfig from a YAML file with "pipeline", "input_dir",
    "output_dir" and the attributes of both config classes.

    Only the attributes of the selected pipeline are used"""
    
    with open(path, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    file_pipeline = content.pop("pipeline", None)
    pipeline = pipeline or file_pipeline

    if pipeline not in PIPELINES:
        raise ValueError(f"{path}: pipeline must be one of {PIPELINES}, got {pipeline}")
    if pipeline == "vision" and pretrained is not None:
        raise ValueError("--pretrained is only used by the clip pipeline")

    known = {f.name for config_class in PIPELINES.values() for f in dataclasses.fields(config_class)} | set(DIR_KEYS)
    unknown = sorted(set(content) - known)

    if unknown:
        raise ValueError(f"{path}: unknown options {unknown}")

    config_class = PIPELINES[pipeline]
    fields = {f.name: f for f in dataclasses.fields(config_class)}
    values = {name: value for name, value in content.items() if name in fields}

    if pipeline == "vision":
        values.update(overrides(model_name=model))
    else:
        values.update(overrides(clip_model=model, clip_pretrained=pretrained))

    missing = sorted((set(fields) - set(values)) | (set(DIR_KEYS) - set(content)))

    if missing:
        raise ValueError(f"{path}: missing options of the {pipeline} pipeline {missing}")

    for name, field in fields.items():
        if field.type is tuple and isinstance(values[name], list):  # YAML lists -> tuples
            values[name] = tuple(values[name])

    dirs = {key: Path(content[key]) for key in DIR_KEYS}  # Relative to the working directory
    dirs.update(overrides(input_dir=input_dir, output_dir=output_dir))
    global_cfg = GlobalConfig(input_dir=dirs["input_dir"], output_dir=dirs["output_dir"])

    return pipeline, config_class(**values), global_cfg


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML file with the pipeline and its options (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--pipeline",
        choices=tuple(PIPELINES),
        default=None,
        help="Override the pipeline of the config file. vision: ResNet / ViT; clip: CLIP zero-shot + WiSE-FT",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model of the config file. vision: model_name; clip: clip_model (OpenCLIP architecture)",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="clip only: override clip_pretrained (OpenCLIP weights tag of the model, e.g. dfn2b, openai)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override input_dir of the config file: folder with train.npz, validation.npz and test.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output_dir of the config file: folder of the run folders"
    )

    return parser, parser.parse_args(argv)


def overrides(**options):
    return {name: value for name, value in options.items() if value is not None}


def main(argv=None):
    parser, args = parse_args(argv)

    # The config is built (and validated) before loading the data
    try:
        pipeline, cfg, global_cfg = load_config(
            args.from_config,
            args.pipeline,
            model=args.model,
            pretrained=args.pretrained,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    setup_logging()
    logger.info(f"config {args.from_config} | pipeline {pipeline}")
    device, amp, use_grad_scaler = setup_device()
    logger.info(f"device {device} | data {global_cfg.input_dir} | results {global_cfg.output_dir}")

    # Load the data (train / validation / test)
    data = load_data(global_cfg)

    # Vision pipeline: ResNet / ViT
    if pipeline == "vision":
        return run_vision(global_cfg, cfg, data, device, amp, use_grad_scaler)

    # CLIP pipeline: zero-shot + WiSE-FT
    return run_clip(global_cfg, cfg, data, device, amp)


if __name__ == "__main__":
    main()
