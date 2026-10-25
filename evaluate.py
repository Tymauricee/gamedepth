"""Auswertung eines trainierten Netzes auf dem RHD-Testteil.

  python evaluate.py --rhd /content/data/rhd --checkpoint runs/rhd_pre_aug/last.pt

Schreibt die Ergebnisse als JSON und speichert Beispielbilder (Foto | echte Tiefe | Schätzung).
Den Vergleich mehrerer Modelle samt Depth Anything V2 macht analyze.py.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from gamedepth import data as D
from gamedepth.metrics import evaluate
from gamedepth.model import DepthUNet


def colorize(depth, valid=None):
    """Tiefe als Farbbild: nah = hell, fern = dunkel (pro Bild normiert)."""
    inv = 1.0 / np.maximum(depth, 1e-3)
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    v = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if valid is not None:
        img[~valid] = 0
    return img


def denormalize(t):
    x = t.numpy().transpose(1, 2, 0) * D.IMAGENET_STD + D.IMAGENET_MEAN
    return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhd", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = D.build("rhd", args.rhd, "val", size=320, augment=False, limit=args.limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DepthUNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    res = evaluate(model, DataLoader(ds, 16, num_workers=min(8, os.cpu_count() or 2)), device)
    name = Path(args.checkpoint).parent.name
    # Beispielbilder
    model.eval()
    tiles = []
    with torch.no_grad():
        for i in np.linspace(0, len(ds) - 1, args.examples).astype(int):
            s = ds[i]
            pred = torch.exp(model(s["image"][None].to(device)))[0, 0].cpu().numpy()
            valid = s["valid"][0].numpy()
            tiles.append(np.hstack([denormalize(s["image"]), colorize(s["depth"][0].numpy(), valid), colorize(pred)]))
    cv2.imwrite(str(out / f"{name}_beispiele.jpg"), np.vstack(tiles))

    (out / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(name, json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
