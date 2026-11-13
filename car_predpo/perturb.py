from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_CONFIG = {
    "gaussian_blur": {"radius": 1.2},
    "resize_restore": {"scale": 0.65},
    "jpeg": {"quality": 70},
    "brightness_contrast": {"brightness": 1.05, "contrast": 0.95},
}


def make_perturbations(image: Image.Image, config: dict | None = None):
    cfg = config or DEFAULT_CONFIG
    source = image.convert("RGB")
    width, height = source.size
    outputs = {}
    outputs["gaussian_blur"] = source.filter(ImageFilter.GaussianBlur(cfg["gaussian_blur"]["radius"]))
    scale = cfg["resize_restore"]["scale"]
    small = source.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
    outputs["resize_restore"] = small.resize((width, height), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    source.save(buffer, "JPEG", quality=cfg["jpeg"]["quality"])
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        outputs["jpeg"] = compressed.convert("RGB").copy()
    adjusted = ImageEnhance.Brightness(source).enhance(cfg["brightness_contrast"]["brightness"])
    outputs["brightness_contrast"] = ImageEnhance.Contrast(adjusted).enhance(
        cfg["brightness_contrast"]["contrast"]
    )
    return outputs, {name: dict(params) for name, params in cfg.items()}

