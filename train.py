"""Training des Tiefen-CNN.

Beispiele:
  python train.py --data rhd --root /content/data/rhd --pretrained --aug --out runs/rhd_pre_aug
  python train.py --data gta --root /content/data/gta --pretrained --aug --out runs/gta_pre_aug
  python train.py --data mix --root /content/data/rhd --gta-root /content/data/gta --gta-share 0.3 \
                  --pretrained --aug --out runs/mix30_pre_aug     (70 % RHD, 30 % GTA)
Jede Epoche wird auf dem RHD-Testteil ausgewertet (inklusive Handtest), damit
GTA- und RHD-Training direkt vergleichbar sind (--eval-root).
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gamedepth import data as D
from gamedepth.losses import total_loss
from gamedepth.metrics import evaluate
from gamedepth.model import DepthUNet, count_parameters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["rhd", "gta", "mix"], required=True,
                    help="Trainingsdatensatz; mix = RHD und GTA gemischt")
    ap.add_argument("--root", required=True, help="Ordner des Trainingsdatensatzes (bei mix: RHD)")
    ap.add_argument("--gta-root", default=None, help="nur bei mix: GTA-Ordner")
    ap.add_argument("--gta-share", type=float, default=0.3, help="nur bei mix: Anteil der GTA-Bilder")
    ap.add_argument("--eval-root", default=None, help="RHD-Ordner für Test und Handtest (Standard: --root bei rhd)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256, help="Kantenlänge der Trainingsausschnitte")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-vortrainierter Encoder")
    ap.add_argument("--aug", action="store_true", help="Farb-, Unschärfe- und Rausch-Augmentierung")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                    help="parallele Ladeprozesse; Bilder dekodieren ist oft der Engpass")
    ap.add_argument("--limit", type=int, default=None, help="nur N Bilder (zum schnellen Testen)")
    ap.add_argument("--eval-batches", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)

    sampler = None
    if args.data == "mix":
        if not args.gta_root:
            raise SystemExit("Bei --data mix bitte --gta-root mit dem GTA-Ordner angeben.")
        train_set, sampler = D.build_mix(args.root, args.gta_root, args.size, augment=args.aug,
                                         limit=args.limit, gta_share=args.gta_share)
    else:
        train_set = D.build(args.data, args.root, "train", args.size, augment=args.aug, limit=args.limit)
    eval_root = args.eval_root or (args.root if args.data in ("rhd", "mix") else None)
    if eval_root is None:
        raise SystemExit("Bitte --eval-root mit dem RHD-Ordner angeben (für den Handtest).")
    val_set = D.build("rhd", eval_root, "val", size=320, augment=False, limit=args.limit)
    train_loader = DataLoader(train_set, args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                              pin_memory=device == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch, shuffle=False, num_workers=args.workers)

    model = DepthUNet(pretrained=args.pretrained).to(device)
    print(f"Gerät: {device} | Parameter: {count_parameters(model) / 1e6:.1f} Mio. | "
          f"Training: {len(train_set)} Bilder | Test: {len(val_set)} Bilder")
    if sampler is not None:
        print(f"Gemischt: pro Epoche {sampler.num_samples} gezogene Bilder, "
              f"im Mittel {args.gta_share:.0%} aus GTA und {1 - args.gta_share:.0%} aus RHD")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    log_path = out / "log.csv"
    best = None
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "absrel", "rmse", "delta1", "hand_acc", "hand_base",
                                "hand_ratio_err", "hand_n", "minutes"])
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                pred = model(img)
            loss, parts = total_loss(pred.float(), gt, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item(); n += 1
            if step % 100 == 0:
                print(f"  Epoche {epoch} Schritt {step}/{len(train_loader)} Verlust {loss.item():.4f}", flush=True)

        m = evaluate(model, val_loader, device, max_batches=args.eval_batches)
        minutes = (time.time() - t0) / 60
        print(f"Epoche {epoch}: Verlust {running / max(n, 1):.4f} | AbsRel {m.get('absrel', float('nan')):.3f} | "
              f"δ1 {m.get('delta1', float('nan')):.3f} | Handtest {m.get('hand_acc', float('nan')):.1%} "
              f"(Basisrate {m.get('hand_base', float('nan')):.1%}, Verhältnisfehler "
              f"{m.get('hand_ratio_err', float('nan')):.3f}, n={m.get('hand_n', 0)}) | {minutes:.1f} min", flush=True)
        if device == "cuda":  # Spitzenbedarf an Grafikspeicher, z. B. für den Antrag auf Rechenressourcen
            print(f"  VRAM Spitze: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB belegt, "
                  f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserviert", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(running / max(n, 1), 5), m.get("absrel"), m.get("rmse"),
                                    m.get("delta1"), m.get("hand_acc"), m.get("hand_base"),
                                    m.get("hand_ratio_err"), m.get("hand_n"), round(minutes, 2)])
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": m, "config": vars(args)}
        torch.save(ckpt, out / "last.pt")
        if best is None or m.get("absrel", 9) < best:
            best = m.get("absrel", 9)
            torch.save(ckpt, out / "best.pt")
    print(f"Fertig. Netz der letzten Epoche (für Ergebnisse verwenden): {out / 'last.pt'}")


if __name__ == "__main__":
    main()
