"""Auswertung: Standardmetriken der Tiefenschätzung und der Handtest.

Standardmetriken (nach Median-Skalierung, weil das Netz nur Verhältnisse lernt):
  AbsRel = mittlerer relativer Fehler |pred - gt| / gt          (kleiner = besser)
  RMSE   = Wurzel des mittleren quadratischen Fehlers in Metern (kleiner = besser)
  δ1     = Anteil der Pixel mit max(pred/gt, gt/pred) < 1,25     (größer = besser)

Handtest: Ist die Hand laut Netz näher an der Kamera als der Körper, und stimmt
das mit der echten Reihenfolge überein? Verglichen werden die Mediane der
geschätzten Tiefe in Hand- und Körperregion.
  hand_acc        Anteil richtiger Reihenfolgen
  hand_base       Basisrate: Anteil, den ein Netz erreicht, das IMMER „Hand vorne“ sagt.
                  hand_acc ist nur aussagekräftig, wenn es deutlich darüber liegt.
  hand_ratio_err  mittlerer Fehler des Verhältnisses Handtiefe/Körpertiefe
                  (|log(pred) - log(echt)|, 0 = perfekt; strenger als die Reihenfolge)
"""
import numpy as np
import torch

from .data import BODY, HAND_MIN

MIN_PIXELS = 50  # zu kleine Hand- oder Körperflächen werden nicht gewertet


def depth_metrics(pred_depth, gt_depth, valid):
    """Für ein Bild (numpy, gleiche Form). Gibt dict oder None zurück."""
    p, g = pred_depth[valid], gt_depth[valid]
    if g.size < 100:
        return None
    p = p * (np.median(g) / np.median(p))
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def hand_test(pred_depth, gt_depth, mask):
    """Gibt (richtig, echte_hand_vorne, verhältnisfehler) zurück oder None, wenn nicht auswertbar."""
    hand, body = mask >= HAND_MIN, mask == BODY
    if hand.sum() < MIN_PIXELS or body.sum() < MIN_PIXELS:
        return None
    gt_ratio = np.median(gt_depth[hand]) / np.median(gt_depth[body])
    pred_ratio = np.median(pred_depth[hand]) / np.median(pred_depth[body])
    gt_closer, pred_closer = gt_ratio < 1, pred_ratio < 1
    return bool(pred_closer == gt_closer), bool(gt_closer), float(abs(np.log(pred_ratio) - np.log(gt_ratio)))


def summarize_hands(hand_rows):
    return {
        "hand_acc": float(np.mean([r[0] for r in hand_rows])),
        "hand_base": float(np.mean([r[1] for r in hand_rows])),
        "hand_ratio_err": float(np.mean([r[2] for r in hand_rows])),
        "hand_n": len(hand_rows),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Wertet ein Netz auf einem Datensatz aus. Gibt Mittelwerte zurück."""
    model.eval()
    rows, hand_rows = [], []
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        pred = torch.exp(model(batch["image"].to(device))).float().cpu().numpy()[:, 0]
        gt = batch["depth"].numpy()[:, 0]
        valid = batch["valid"].numpy()[:, 0]
        mask = batch["mask"].numpy()
        for i in range(len(pred)):
            m = depth_metrics(pred[i], gt[i], valid[i])
            if m:
                rows.append(m)
            h = hand_test(pred[i], gt[i], mask[i])
            if h:
                hand_rows.append(h)
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("absrel", "rmse", "delta1")} if rows else {}
    if hand_rows:
        out.update(summarize_hands(hand_rows))
    return out
