"""Vergleich und Fehleranalyse: alle Modelle auf denselben RHD-Testbildern.

  python analyze.py --rhd /content/data/rhd --out eval --da2 \
      --checkpoints runs/rhd_pre_aug/last.pt runs/gta_pre_aug/last.pt

Für jedes Modell:
  1. Standardmaße (AbsRel, RMSE, δ1) und Handtest wie in evaluate.py, dazu
     - dieselben Maße nur auf der Person (Hand + Körper, ohne den Hintergrund, der in RHD
       ein flaches Foto in 5 m Entfernung ist): absrel_person, delta1_person
     - absrel_person_affin, delta1_person_affin: wie oben, aber alle Modelle mit derselben
       Anpassung (Maßstab + Versatz in inverser Tiefe, wie bei Depth Anything). Nur so ist
       der Vergleich mit Depth Anything fair, denn die eigenen Netze bekommen sonst nur
       einen Maßstab (Median-Skalierung), Depth Anything aber Maßstab und Versatz.
     - trennschaerfe: Mittel aus der Trefferquote bei Hand vorne und bei Hand hinten
       (balanced accuracy). Raten oder immer dieselbe Antwort ergibt 0,5, egal wie
       oft die Hand vorne ist; deshalb fairer als hand_acc mit Basisrate.
  2. Test der Hypothese „weiter unten im Bild = näher“ (van Dijk & de Croon, ICCV 2019).
     Die Testbilder werden in vier Gruppen geteilt: Hand im Bild tiefer oder höher
     als der Körper, Hand wirklich vorne oder hinten. Daraus zwei Kennzahlen:
       einfluss_wahrheit  Wie stark ändert sich die Antwort „Hand vorne“, wenn die Hand
                          wirklich vorne ist? (bei gleicher Bildhöhe)
       einfluss_hoehe     Wie stark ändert sie sich, wenn die Hand im Bild tiefer liegt?
                          (bei gleicher Wahrheit)
     Ein gutes Modell: Wahrheit nahe 1, Höhe nahe 0. Ein Modell, das nur nach der
     Bildhöhe schätzt: Wahrheit nahe 0, Höhe nahe 1.

Mit --baseline kommt eine flache Schätzung dazu (überall dieselbe Tiefe). Sie zeigt, was ein
Modell ohne jedes Wissen schon erreicht; gerade auf der Person ist das nicht wenig, weil dort
alle Tiefen nah beieinander liegen.

Depth Anything V2 Small wird von Hugging Face geladen und läuft auf der GPU. Es liefert
relative inverse Tiefe (groß = nah). Für Standardmaße und Verhältnisfehler wird sie pro
Bild mit Maßstab und Versatz an die echte inverse Tiefe angepasst und auf höchstens 5 m
begrenzt, wie in der MiDaS-Auswertung üblich. Die Reihenfolge Hand/Körper wird ohne
Anpassung direkt aus der Modellausgabe gelesen.

Schreibt analyse.json in den Ordner --out und gibt eine Tabelle aus.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gamedepth import data as D
from gamedepth.metrics import MIN_PIXELS, depth_metrics
from gamedepth.model import DepthUNet

DA2_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MAX_DEPTH = D.MAX_DEPTH["rhd"]


# ---------------------------------------------------------------- Modelle
def net_predictor(path, device):
    """Eigenes Netz: Ausgabe log(Tiefe) -> Tiefe. Gilt für Maße und Reihenfolge."""
    model = DepthUNet(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device)["model"])

    def predict(img):
        return torch.exp(model(img.to(device))).float()[:, 0].cpu().numpy()
    return predict


def flat_predictor():
    """Vergleichszeile ohne jedes Wissen: überall dieselbe Tiefe."""
    def predict(img):
        return np.ones((img.shape[0], img.shape[2], img.shape[3]), np.float32)
    return predict


def da2_predictor(device, model_id=DA2_ID):
    """Depth Anything V2: Ausgabe relative inverse Tiefe in Eingabegröße."""
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def predict(img):
        x = F.interpolate(img.to(device), size=(518, 518), mode="bicubic", align_corners=False)
        disp = model(pixel_values=x).predicted_depth[:, None]
        disp = F.interpolate(disp, size=img.shape[-2:], mode="bilinear", align_corners=False)
        return disp[:, 0].float().cpu().numpy()
    return predict


def align_inverse(disp, gt, valid):
    """Maßstab und Versatz per Ausgleichsrechnung im Raum der inversen Tiefe,
    danach auf höchstens MAX_DEPTH begrenzt (sonst entstehen Tiefen von 1000 m)."""
    A = np.stack([disp[valid], np.ones(valid.sum())], 1)
    (a, b), *_ = np.linalg.lstsq(A, 1.0 / gt[valid], rcond=None)
    return 1.0 / np.maximum(a * disp + b, 1.0 / MAX_DEPTH)


# ---------------------------------------------------------------- Handtest mit Bildhöhe
def hand_record(pred_metric, pred_order, gt, mask):
    hand, body = mask >= D.HAND_MIN, mask == D.BODY
    if hand.sum() < MIN_PIXELS or body.sum() < MIN_PIXELS:
        return None
    rows = np.arange(mask.shape[0])[:, None]
    hand_y = (rows * hand).sum() / hand.sum()
    body_y = (rows * body).sum() / body.sum()
    gt_ratio = np.median(gt[hand]) / np.median(gt[body])
    pred_ratio = np.median(pred_metric[hand]) / np.median(pred_metric[body])
    return {
        "gt": bool(gt_ratio < 1),
        "pred": bool(np.median(pred_order[hand]) < np.median(pred_order[body])),
        "lower": bool(hand_y > body_y),  # Bildzeilen zählen von oben nach unten
        "ratio_err": float(abs(np.log(pred_ratio) - np.log(gt_ratio))),
    }


def nanmean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def summarize(recs):
    gt = np.array([r["gt"] for r in recs])
    pred = np.array([r["pred"] for r in recs])
    lower = np.array([r["lower"] for r in recs])

    def share(sel):  # Anteil „Hand vorne“ unter den ausgewählten Bildern
        return float(pred[sel].mean()) if sel.sum() >= 10 else float("nan")

    groups = {f"hand_{'tiefer' if lo else 'hoeher'}_{'vorne' if g else 'hinten'}": (lo, g)
              for lo in (True, False) for g in (True, False)}
    cells = {k: {"n": int(((lower == lo) & (gt == g)).sum()), "sagt_hand_vorne": share((lower == lo) & (gt == g))}
             for k, (lo, g) in groups.items()}
    truth = nanmean([share((lower == lo) & gt) - share((lower == lo) & ~gt) for lo in (True, False)])
    height = nanmean([share(lower & (gt == g)) - share(~lower & (gt == g)) for g in (True, False)])
    hit_front, hit_back = float(pred[gt].mean()), float((~pred[~gt]).mean())
    return {
        "hand_acc": float((pred == gt).mean()),
        "hand_base": float(gt.mean()),
        "treffer_hand_vorne": hit_front,
        "treffer_hand_hinten": hit_back,
        "trennschaerfe": (hit_front + hit_back) / 2,
        "hand_ratio_err": float(np.mean([r["ratio_err"] for r in recs])),
        "hand_n": len(recs),
        "sagt_hand_vorne": float(pred.mean()),
        "hoehenregel_acc": float((lower == gt).mean()),
        "einfluss_wahrheit": float(truth),
        "einfluss_hoehe": float(height),
        "gruppen": cells,
    }


# ---------------------------------------------------------------- Ablauf
def analyze(name, predict, loader, aligned):
    rows, rows_p, rows_pa, recs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = predict(batch["image"])
            gts, valids, masks = batch["depth"].numpy()[:, 0], batch["valid"].numpy()[:, 0], batch["mask"].numpy()
            for raw, gt, valid, mask in zip(out, gts, valids, masks):
                if aligned:  # Depth Anything V2: raw ist inverse Tiefe
                    metric, order = align_inverse(raw, gt, valid), 1.0 / np.maximum(raw, 1e-6)
                else:
                    metric = order = raw
                m = depth_metrics(metric, gt, valid)
                if m:
                    rows.append(m)
                person = valid & (mask >= D.BODY)  # nur Körper und Hand, ohne Hintergrund
                m = depth_metrics(align_inverse(raw, gt, person) if aligned else raw, gt, person)
                if m:
                    rows_p.append(m)
                disp = raw if aligned else 1.0 / np.maximum(raw, 1e-6)  # gleiche Anpassung für alle
                m = depth_metrics(align_inverse(disp, gt, person), gt, person)
                if m:
                    rows_pa.append(m)
                r = hand_record(metric, order, gt, mask)
                if r:
                    recs.append(r)
    res = {k: float(np.mean([r[k] for r in rows])) for k in ("absrel", "rmse", "delta1")}
    res.update({f"{k}_person": float(np.mean([r[k] for r in rows_p])) for k in ("absrel", "rmse", "delta1")})
    res.update({f"{k}_person_affin": float(np.mean([r[k] for r in rows_pa])) for k in ("absrel", "rmse", "delta1")})
    res.update(summarize(recs))
    print(f"{name}: fertig ({len(rows)} Bilder)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhd", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--da2", action="store_true", help="Depth Anything V2 Small mit auswerten")
    ap.add_argument("--da2-model", default=DA2_ID)
    ap.add_argument("--baseline", action="store_true", help="flache Schätzung als Vergleichszeile")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = D.build("rhd", args.rhd, "val", size=320, augment=False, limit=args.limit)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=min(8, os.cpu_count() or 2))
    print(f"Gerät: {device} | Testbilder: {len(ds)}", flush=True)

    results = {}
    for path in args.checkpoints:
        if not Path(path).exists():
            print(f"Übersprungen, Datei fehlt: {path}", flush=True)
            continue
        name = Path(path).parent.name
        results[name] = analyze(name, net_predictor(path, device), loader, aligned=False)
    if args.baseline:
        results["flache_schaetzung"] = analyze("flache_schaetzung", flat_predictor(), loader, aligned=False)
    if args.da2:
        results["depth_anything_v2_small"] = analyze("depth_anything_v2_small", da2_predictor(device, args.da2_model),
                                                     loader, aligned=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(results).replace("NaN", "null"))  # NaN ist kein gültiges JSON
    (out / "analyse.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False))

    cols = ["absrel", "absrel_person", "absrel_person_affin", "delta1_person_affin", "hand_acc", "trennschaerfe",
            "hand_ratio_err", "einfluss_wahrheit", "einfluss_hoehe", "hand_n"]
    print("\n" + "modell".ljust(26) + "".join(c.rjust(21) for c in cols))
    for name, r in results.items():
        print(name.ljust(26) + "".join(f"{r[c]:21.3f}" if c != "hand_n" else f"{r[c]:21d}" for c in cols))
    any_r = next(iter(results.values()), None)
    if any_r:
        print(f"\nHöhenregel („tiefer im Bild = näher“) wäre auf diesen Bildern zu {any_r['hoehenregel_acc']:.1%} richtig.")
        print("Gruppen (Anzahl Bilder | Anteil „Hand vorne“ je Modell):")
        for g, c in any_r["gruppen"].items():
            line = "  ".join(f"{n}: {r['gruppen'][g]['sagt_hand_vorne']:.0%}" for n, r in results.items())
            print(f"  {g:24s} n={c['n']:5d} | {line}")
    print(f"\nGespeichert: {out / 'analyse.json'}")


if __name__ == "__main__":
    main()
