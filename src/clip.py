from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .utils import safe_tag

# ----- Prompt templates -----

# CLIP prompt templates (openai/CLIP: data/prompts.md for CIFAR-10, Prompt_Engineering_for_ImageNet.ipynb for ImageNet)
CIFAR_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

PROMPT_SETS = {
    "base": ["a photo of a {}."],
    "cifar": CIFAR_TEMPLATES,
    "imagenet": IMAGENET_TEMPLATES,
    "cifar+imagenet": CIFAR_TEMPLATES + [t for t in IMAGENET_TEMPLATES if t not in CIFAR_TEMPLATES],
}


# Values come from configs/config.yaml (main.load_config); no defaults
@dataclass
class CLIPConfig:
    clip_model: str
    clip_pretrained: str
    prompt_set: str
    interpolation: str
    tta_flip: bool
    batch_size: int
    wise_alpha: float  # None: zero-shot; a: W = (1 - a) W_zeroshot + a W_trained, b = a b_trained
    epochs: int
    batch_size_head: int
    optimizer: str
    scheduler: str  # "cosine", "constant", "onecycle" or None (no scheduler)
    warmup_fraction: float
    lr: float
    weight_decay: float
    l2sp: float  # Towards the zero-shot head
    label_smoothing: float
    save_best_val_checkpoint: bool
    features_from: str  # Experiment folder: reuse its cache/ features
    seed: int

    @property
    def tag(self):
        return safe_tag(self.clip_model, self.clip_pretrained)

    @property
    def prompt_tag(self):
        return self.prompt_set.replace("+", "-")

    @property
    def zeroshot_name(self):
        return f"clip_zeroshot_{self.tag}_prompt_{self.prompt_tag}"

    @property
    def name(self):
        if self.wise_alpha is None:
            return self.zeroshot_name

        return f"clip_wise_a{round(self.wise_alpha * 100):03d}_{self.tag}_prompt_{self.prompt_tag}"

    def feature_file(self, split):
        return safe_tag("clipfeat", self.tag, split, self.interpolation, f"tta{int(self.tta_flip)}")


# Load CLIP model and tokenizer
def load_clip(cfg, device):
    """OpenCLIP model on device + tokenizer, input size, normalization and logit scale"""
    name = cfg.clip_model

    # Adjust name
    if not open_clip.pretrained.get_pretrained_cfg(name, cfg.clip_pretrained) and name.endswith("-quickgelu"):
        name = name[: -len("-quickgelu")]

    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=cfg.clip_pretrained, force_quick_gelu=cfg.clip_model.endswith("-quickgelu")
    )
    model = model.to(device).eval()

    normalize = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))

    size = model.visual.image_size
    info = {
        "model": cfg.clip_model,
        "loaded_as": name,
        "pretrained": cfg.clip_pretrained,
        "input_size": size[0] if isinstance(size, (tuple, list)) else size,
        "mean": tuple(normalize.mean),
        "std": tuple(normalize.std),
        "logit_scale": model.logit_scale.exp().item(),
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, open_clip.get_tokenizer(name), info


@torch.no_grad()
def text_classifier(model, tokenizer, templates, class_names, device, amp):
    """Class embeddings (classes, d): mean of the L2-normalized prompt embeddings, L2-normalized"""
    weights = []

    for name in class_names:
        tokens = tokenizer([template.format(name) for template in templates]).to(device)

        with torch.autocast(**amp):
            embeddings = F.normalize(model.encode_text(tokens).float(), dim=-1)

        weights.append(F.normalize(embeddings.mean(0), dim=0))

    return torch.stack(weights).cpu()
