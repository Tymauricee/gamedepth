"""Wie groß und wie schnell? Eigenes Netz gegen Depth Anything V2 Small auf demselben Rechner.

  python speed_test.py --checkpoint rhd.pt --da2

Gemessen wird nur der Durchlauf durch das Netz (ohne Kamera, Umrechnen und Einfärben), jeweils
mit der Eingabegröße, die auch in Demo und Auswertung benutzt wird: eigenes Netz 320 × 256,
Depth Anything 518 × 518. Erst einige Aufwärmdurchläufe, dann der Mittelwert über viele
Durchläufe; auf der Grafikkarte mit torch.cuda.synchronize(), damit wirklich die Rechenzeit
gemessen wird und nicht nur das Abschicken der Arbeit. Zusätzlich auf dem Prozessor (CPU), als
grober Hinweis auf Geräte ohne starke Grafikkarte.
"""
import argparse
import time

import torch

from gamedepth.model import DepthUNet

DA2_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def own_net(path):
    model = DepthUNet(pretrained=False).eval()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model, (256, 320), lambda m, x: m(x)


def da2_net(model_id):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval()
    return model, (518, 518), lambda m, x: m(pixel_values=x).predicted_depth


def ms_per_image(model, forward, size, device, runs, warmup):
    model = model.to(device)
    x = torch.randn(1, 3, *size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            forward(model, x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            forward(model, x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="eigenes Netz, z. B. rhd.pt")
    ap.add_argument("--da2", action="store_true", help="Depth Anything V2 Small mitmessen")
    ap.add_argument("--da2-model", default=DA2_ID)
    ap.add_argument("--runs", type=int, default=100, help="Durchläufe auf der Grafikkarte")
    ap.add_argument("--cpu-runs", type=int, default=10, help="Durchläufe auf dem Prozessor (0 = weglassen)")
    args = ap.parse_args()

    models = {}
    if args.checkpoint:
        models["eigenes Netz"] = own_net(args.checkpoint)
    if args.da2:
        models["Depth Anything V2 Small"] = da2_net(args.da2_model)
    if not models:
        raise SystemExit("Nichts zu messen: --checkpoint und/oder --da2 angeben")

    gpu = torch.cuda.is_available()
    print(f"Grafikkarte: {torch.cuda.get_device_name(0) if gpu else 'keine'} | "
          f"Prozessor-Threads: {torch.get_num_threads()} | PyTorch {torch.__version__}\n")
    rows = []
    for name, (model, size, forward) in models.items():
        params = sum(p.numel() for p in model.parameters()) / 1e6
        g = ms_per_image(model, forward, size, "cuda", args.runs, 10) if gpu else float("nan")
        c = ms_per_image(model, forward, size, "cpu", args.cpu_runs, 2) if args.cpu_runs else float("nan")
        rows.append((name, params, size, g, c))
        print(f"{name}: fertig", flush=True)

    print(f"\n{'Modell':26s}{'Parameter':>12s}{'Eingabe':>12s}{'GPU ms/Bild':>14s}{'GPU Bilder/s':>14s}"
          f"{'CPU ms/Bild':>14s}{'CPU Bilder/s':>14s}")
    for name, params, (h, w), g, c in rows:
        print(f"{name:26s}{params:10.1f} M{f'{w}x{h}':>12s}{g:14.1f}{1000 / g:14.1f}{c:14.0f}{1000 / c:14.1f}")
    if len(rows) == 2:
        (_, p1, _, g1, c1), (_, p2, _, g2, c2) = rows
        print(f"\nDepth Anything hat {p2 / p1:.1f}-mal so viele Parameter und braucht auf der Grafikkarte "
              f"{g2 / g1:.1f}-mal, auf dem Prozessor {c2 / c1:.1f}-mal so lange pro Bild.")


if __name__ == "__main__":
    main()
