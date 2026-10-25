"""Prüft einen Datensatz, bevor du trainierst.

  python check_data.py --data rhd --root /content/data/rhd

Gibt Tiefenstatistiken aus, prüft bei RHD die Kodierung (Hand meist vor dem
Körper, Werte zwischen 0 und 5 m) und speichert ein Übersichtsbild.
"""
import argparse

import cv2
import numpy as np

from gamedepth import data as D
from evaluate import colorize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["rhd", "gta"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="datencheck.jpg")
    args = ap.parse_args()

    ds = D.build(args.data, args.root, "train", size=320, augment=False)
    print(f"{len(ds)} Trainingsbilder gefunden")
    idx = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)
    depths, hand_closer, tiles = [], [], []
    per_class = {"Hintergrund": [], "Person": [], "Hand": []}
    for k, i in enumerate(idx):
        rgb, depth, mask = ds.load(i)
        depths.append(depth[np.isfinite(depth)])
        if mask is not None:
            for name, sel in [("Hintergrund", mask == 0), ("Person", mask == 1), ("Hand", mask >= 2)]:
                if sel.sum() > 50:
                    per_class[name].append(np.median(depth[sel]))
            if (mask >= 2).sum() > 50 and (mask == 1).sum() > 50:
                hand_closer.append(np.median(depth[mask >= 2]) < np.median(depth[mask == 1]))
        if k < 6:
            vis = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 320))
            tiles.append(np.hstack([vis, cv2.resize(colorize(depth), (320, 320))]))
    allv = np.concatenate(depths)
    unit = "m" if args.data == "rhd" else "Spieleinheiten"
    print(f"Tiefe: min {allv.min():.3f} | Median {np.median(allv):.2f} | max {allv.max():.2f} ({unit})")
    if args.data == "gta":
        print(f"Anteil Pixel ab 1000 (Himmel/Fernbereich, wird nicht trainiert): {np.mean(allv >= 1000):.1%}")
    for name, vals in per_class.items():
        if vals:
            print(f"  {name:12s} Median-Tiefe {np.median(vals):.2f} m (aus {len(vals)} Bildern)")
    if hand_closer:
        share = np.mean(hand_closer)
        print(f"Hand näher als Körper (echte Tiefe): {share:.1%} von {len(hand_closer)} Bildern")
        if args.data == "rhd" and not (0.3 <= np.median(allv) <= 5.0):
            print("WARNUNG: Tiefenwerte unplausibel – Kodierung in data.rhd_decode_depth prüfen!")
    cv2.imwrite(args.out, np.vstack(tiles))
    print(f"Übersichtsbild: {args.out}")


if __name__ == "__main__":
    main()
