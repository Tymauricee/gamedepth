"""Verlustfunktion: skaleninvarianter Log-Fehler (Eigen et al. 2014) + Kanten-Term.

Warum skaleninvariant? Aus einem einzelnen Bild ist die absolute Größe nicht
eindeutig: Eine Spielzeughand nah an der Kamera sieht aus wie eine echte Hand
weiter weg. Außerdem haben Blender-Kamera und Webcam verschiedene Brennweiten.
Der Verlust bestraft deshalb nicht einen gemeinsamen Faktor für das ganze Bild,
sondern nur falsche *Verhältnisse*, also ob die Hand näher ist als der Körper.
"""
import torch
import torch.nn.functional as F


def scale_invariant_loss(pred_log, gt_depth, valid, lam=0.85):
    d = (pred_log - torch.log(gt_depth)) * valid
    n = valid.sum(dim=(1, 2, 3)).clamp(min=1)
    mean_d = d.sum(dim=(1, 2, 3)) / n
    mean_d2 = (d ** 2).sum(dim=(1, 2, 3)) / n
    return torch.sqrt((mean_d2 - lam * mean_d ** 2).clamp(min=1e-8)).mean()


def gradient_loss(pred_log, gt_depth, valid, scales=3):
    """Vergleicht die Änderungen der Tiefe zwischen Nachbarpixeln: hält Kanten scharf."""
    total = 0.0
    d = (pred_log - torch.log(gt_depth)) * valid
    v = valid.float()
    for s in range(scales):
        if s > 0:
            d = F.avg_pool2d(d, 2)
            v = (F.avg_pool2d(v, 2) > 0.99).float()
        gx = (d[..., :, 1:] - d[..., :, :-1]).abs() * v[..., :, 1:] * v[..., :, :-1]
        gy = (d[..., 1:, :] - d[..., :-1, :]).abs() * v[..., 1:, :] * v[..., :-1, :]
        total = total + (gx.sum() + gy.sum()) / v.sum().clamp(min=1)
    return total / scales


def total_loss(pred_log, gt_depth, valid, edge_weight=0.5):
    valid = valid.float()
    si = scale_invariant_loss(pred_log, gt_depth, valid)
    gr = gradient_loss(pred_log, gt_depth, valid)
    return si + edge_weight * gr, {"si": si.item(), "grad": gr.item()}
