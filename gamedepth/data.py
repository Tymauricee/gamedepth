"""Datensätze: Bild + Tiefenkarte (+ Maske) als PyTorch-Dataset.

RHD  (Rendered Hand Pose Dataset, Uni Freiburg): 320×320, gerendert mit Blender.
     Tiefe ist in den Kanälen R und G eines PNG kodiert (16 Bit, 0…5 m).
     Maske: 0 = Hintergrund, 1 = Person, 2…33 = Teile der linken/rechten Hand.
GTA  (MVS-Synth aus GTA V): 960×540, Tiefe als EXR. Die Einheit ist nicht
     dokumentiert (nicht Meter: Median ca. 178). Werte ab 1000 (Himmel, Fernbereich)
     werden wie in DUSt3R als ungültig behandelt. Weil der Verlust
     skaleninvariant ist, spielt die Einheit für das Training keine Rolle.
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # vor dem Import von cv2 setzen

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

MAX_DEPTH = {"rhd": 5.0, "gta": 1000.0}  # RHD in Metern, GTA in Spieleinheiten
INVALID_ABOVE = {"rhd": None, "gta": 1000.0}  # ab diesem Wert nicht trainieren
BODY, HAND_MIN = 1, 2                     # RHD-Maskenwerte


# ---------------------------------------------------------------- RHD
def find_rhd_root(path: str) -> Path:
    """Sucht den Ordner, der training/color enthält (z. B. RHD_published_v2)."""
    p = Path(path)
    for cand in [p, *p.glob("*"), *p.glob("*/*")]:
        if (cand / "training" / "color").is_dir():
            return cand
    raise FileNotFoundError(f"Kein RHD-Ordner mit training/color unter {path} gefunden")


def rhd_decode_depth(png_bgr: np.ndarray) -> np.ndarray:
    """RHD-Tiefe: 16-Bit-Wert aus Rot (obere 8 Bit) und Grün (untere 8 Bit), skaliert auf 0…5 m."""
    red = png_bgr[..., 2].astype(np.float32)
    green = png_bgr[..., 1].astype(np.float32)
    return (red * 256.0 + green) / 65535.0 * 5.0


class RHD(Dataset):
    def __init__(self, root, split="training", size=256, augment=False, limit=None):
        self.root = find_rhd_root(root) / split
        self.files = sorted((self.root / "color").glob("*.png"))
        if limit:
            self.files = self.files[:limit]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.files)

    def load(self, i):
        name = self.files[i].name
        rgb = cv2.cvtColor(cv2.imread(str(self.root / "color" / name)), cv2.COLOR_BGR2RGB)
        depth = rhd_decode_depth(cv2.imread(str(self.root / "depth" / name), cv2.IMREAD_UNCHANGED))
        mask = cv2.imread(str(self.root / "mask" / name), cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return rgb, depth, mask

    def __getitem__(self, i):
        rgb, depth, mask = self.load(i)
        return to_sample(rgb, depth, mask, self.size, self.augment, MAX_DEPTH["rhd"])


# ---------------------------------------------------------------- GTA V (MVS-Synth)
def read_exr(path) -> np.ndarray:
    """Liest eine EXR-Tiefenkarte. Erst OpenCV, sonst die OpenEXR-Bibliothek
    (manche OpenCV-Versionen haben keine EXR-Unterstützung)."""
    try:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        d = None
    if d is None:
        import OpenEXR
        with OpenEXR.File(str(path)) as f:
            d = next(iter(f.channels().values())).pixels
    d = np.asarray(d, dtype=np.float32)
    return d[..., 0] if d.ndim == 3 else d


class MVSSynth(Dataset):
    def __init__(self, root, split="training", size=256, augment=False, limit=None, val_sequences=10):
        root = Path(root)
        if not any(root.glob("*/images")):
            root = next(p for p in root.glob("*") if any(p.glob("*/images")))
        seqs = sorted(p for p in root.iterdir() if (p / "images").is_dir())
        k = min(val_sequences, max(1, len(seqs) // 5))  # die letzten k Sequenzen dienen zur Kontrolle
        seqs = seqs[:-k] if split == "training" else seqs[-k:]
        self.files = [f for s in seqs for f in sorted((s / "images").glob("*.png"))]
        if limit:
            self.files = self.files[:limit]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.files)

    def load(self, i):
        f = self.files[i]
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        depth = read_exr(f.parent.parent / "depths" / (f.stem + ".exr"))
        depth[~np.isfinite(depth)] = np.inf  # Himmel: wird in to_sample ungültig
        # auf kurze Seite = 320 verkleinern (spart Rechenzeit, gleiche Größenordnung wie RHD)
        h, w = depth.shape
        s = 320 / min(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST)
        return rgb, depth, None

    def __getitem__(self, i):
        rgb, depth, _ = self.load(i)
        return to_sample(rgb, depth, None, self.size, self.augment, MAX_DEPTH["gta"], INVALID_ABOVE["gta"])


# ---------------------------------------------------------------- gemeinsame Vorverarbeitung
def augment_colors(rgb: np.ndarray) -> np.ndarray:
    """Farb-, Unschärfe- und Rausch-Augmentierung: macht das Netz robuster gegen
    die Unterschiede zwischen gerendertem Bild und echter Webcam."""
    x = rgb.astype(np.float32) / 255.0
    x = x * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1)          # Helligkeit, Kontrast
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * random.uniform(0.6, 1.4)                       # Sättigung
    x = x * np.array([random.uniform(0.9, 1.1) for _ in range(3)], np.float32)  # Farbstich
    if random.random() < 0.4:
        k = random.choice([3, 5])
        x = cv2.GaussianBlur(x, (k, k), 0)                                 # Unschärfe
    if random.random() < 0.5:
        x = x + np.random.normal(0, random.uniform(0.005, 0.03), x.shape).astype(np.float32)  # Sensorrauschen
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def to_sample(rgb, depth, mask, size, augment, max_depth, invalid_above=None):
    h, w = depth.shape
    # zufälliger Ausschnitt beim Training, mittiger Ausschnitt sonst
    if augment:
        y0, x0 = random.randint(0, h - size), random.randint(0, w - size)
    else:
        y0, x0 = (h - size) // 2, (w - size) // 2
    sl = (slice(y0, y0 + size), slice(x0, x0 + size))
    rgb, depth = rgb[sl], depth[sl]
    mask = mask[sl] if mask is not None else np.zeros_like(depth, np.uint8)
    if augment:
        if random.random() < 0.5:
            rgb, depth, mask = rgb[:, ::-1], depth[:, ::-1], mask[:, ::-1]
        rgb = augment_colors(np.ascontiguousarray(rgb))
    valid = (depth > 1e-3) & np.isfinite(depth)
    if invalid_above is not None:
        valid &= depth < invalid_above
    depth = np.clip(depth, 1e-3, max_depth)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return {
        "image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
        "depth": torch.from_numpy(np.ascontiguousarray(depth))[None].float(),
        "valid": torch.from_numpy(np.ascontiguousarray(valid))[None],
        "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
    }


def build_mix(rhd_root, gta_root, size=256, augment=False, limit=None, gta_share=0.3):
    """Gemischtes Training: RHD (Blender) und GTA in einem Datensatz.

    Gibt den Datensatz und einen Sampler zurück. Der Sampler zieht pro Epoche so viele
    Bilder, wie RHD allein hätte (gleiche Rechenzeit wie ein reiner RHD-Lauf), davon im
    Mittel den Anteil gta_share aus GTA. Möglich ist das Mischen, weil der Verlust
    skaleninvariant ist: Meter (RHD) und Spieleinheiten (GTA) stören sich nicht.
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
    rhd = RHD(rhd_root, "training", size, augment, limit)
    gta = MVSSynth(gta_root, "training", size, augment, limit)
    weights = [(1 - gta_share) / len(rhd)] * len(rhd) + [gta_share / len(gta)] * len(gta)
    sampler = WeightedRandomSampler(weights, num_samples=len(rhd), replacement=True)
    return ConcatDataset([rhd, gta]), sampler


def build(name, root, split, size=256, augment=False, limit=None):
    if name == "rhd":
        return RHD(root, "training" if split == "train" else "evaluation", size, augment, limit)
    if name == "gta":
        return MVSSynth(root, "training" if split == "train" else "validation", size, augment, limit)
    raise ValueError(f"Unbekannter Datensatz: {name}")
