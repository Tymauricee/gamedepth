"""Live-Demo für die KI-CON: Webcam-Bild -> Tiefenkarte in Echtzeit.

  python live_demo.py --checkpoints rhd.pt mix.pt gta.pt --da2

Nebeneinander: Kamerabild, Schätzung des eigenen Netzes und (mit --da2) Depth Anything V2
Small zum Vergleich. Hell = nah, dunkel = fern. Besucher sollen die Hand vor den Körper
halten (wird hell) und dann hinter sich auf Hüfthöhe (dort irrt sich das eigene Netz oft,
siehe Webcam-Test).

Tasten (im Demo-Fenster):
  1, 2, 3 …  eigenes Netz wechseln (Reihenfolge wie bei --checkpoints)
  d          Depth Anything ein/aus
  f          Vollbild ein/aus
  s          Bildschirmfoto speichern
  q / Esc    beenden (oder Fenster schließen)

--offline lädt Depth Anything nur aus dem Zwischenspeicher (vorher einmal mit Internet
starten), damit die Demo ohne Netz auf der KI-CON nicht hängt.
"""
import argparse
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gamedepth.data import IMAGENET_MEAN, IMAGENET_STD
from gamedepth.model import DepthUNet

PANEL_W, PANEL_H = 480, 384          # Anzeigegröße je Bild (5:4, wie die Netz-Eingabe)
ASPECT = PANEL_W / PANEL_H
BG, FG, GREY = (24, 24, 24), (240, 240, 240), (170, 170, 170)
NAMES = {"rhd": "RHD-Netz (Blender)", "mix": "Mischnetz (30 % GTA)", "gta": "GTA-Netz"}
FONTS = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]


# ---------------------------------------------------------------- Anzeige
@lru_cache(maxsize=8)
def font(size):
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def bar(width, height, items):
    """Leiste mit Text (mit Umlauten, deshalb über Pillow). items: (x, text, Größe, Farbe RGB)."""
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    for x, text, size, color in items:
        f = font(size)
        top, bottom = f.getbbox("ÄgHy")[1], f.getbbox("ÄgHy")[3]
        if x < 0:  # rechtsbündig
            x = width + x - draw.textlength(text, font=f)
        draw.text((x, (height - (bottom - top)) / 2 - top), text, font=f, fill=color)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=4)
def legend(width):
    """Farbskala und Aufforderung an Besucher (ändert sich nie, deshalb zwischengespeichert)."""
    big = width >= 1400
    out = np.vstack([
        bar(width, 44, [(250, "Probier's aus: Halte die Hand vor deinen Körper – und dann hinter dich auf Hüfthöhe.",
                         20 if big else 14, FG)]),
        bar(width, 26, [(250, "1/2/3 Netz wechseln · D Depth Anything ein/aus · F Vollbild · S Foto · Q Ende",
                         14 if big else 12, GREY)]),
    ])
    ramp = np.linspace(255, 0, 200).astype(np.uint8)[None].repeat(18, 0)  # links hell = nah
    out[12:30, 20:220] = cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO)
    out[34:52, 20:220] = bar(200, 18, [(0, "nah", 13, FG), (-1, "fern", 13, FG)])
    return out


def title(width, fps):
    return bar(width, 56, [(16, "GameDepth – Tiefe aus einem einzigen Kamerabild, gelernt nur aus Spieldaten",
                            24 if width >= 1400 else 17, FG),
                           (-16, f"{fps:4.1f} Bilder/s", 16, GREY)])


def header(width, labels):
    return bar(width, 36, [(i * PANEL_W + 12, text, 17, FG) for i, text in enumerate(labels)])


class Colorizer:
    """Nähe -> Farbe. Hell/dunkel richtet sich nach dem Bild selbst (2. und 98. Perzentil); die
    Grenzen werden über die Zeit geglättet, damit das Bild nicht flackert."""

    def __init__(self, alpha=0.2):
        self.lo = self.hi = None
        self.alpha = alpha

    def __call__(self, near):
        lo, hi = np.percentile(near, 2), np.percentile(near, 98)
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.alpha * (lo - self.lo)
            self.hi += self.alpha * (hi - self.hi)
        v = np.clip((near - self.lo) / max(self.hi - self.lo, 1e-6), 0, 1)
        img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- Modelle
def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def crop_to_panel(frame):
    """Mitte des Kamerabilds auf das Seitenverhältnis 5:4 zuschneiden (keine Verzerrung)."""
    h, w = frame.shape[:2]
    if w / h > ASPECT:
        nw = int(h * ASPECT)
        return frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
    nh = int(w / ASPECT)
    return frame[(h - nh) // 2:(h - nh) // 2 + nh]


def load_own(path, device):
    model = DepthUNet(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model


def load_da2(model_id, device, offline):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"  # nur Zwischenspeicher, kein Internet nötig
    from transformers import AutoModelForDepthEstimation
    return AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()


def window_open(win):
    """False, sobald das Fenster mit dem X geschlossen wurde (sonst würde imshow es neu öffnen)."""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


# ---------------------------------------------------------------- Ablauf
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=[], help="eigene Netze, z. B. rhd.pt mix.pt gta.pt")
    ap.add_argument("--checkpoint", help="ein einzelnes eigenes Netz (wie früher)")
    ap.add_argument("--da2", action="store_true", help="Depth Anything V2 Small zum Vergleich daneben")
    ap.add_argument("--da2-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--da2-size", default="518x420", help="Eingabe Depth Anything, Breite x Höhe (Vielfache von 14)")
    ap.add_argument("--offline", action="store_true", help="Depth Anything nur aus dem Zwischenspeicher laden")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=320, help="Eingabebreite eigenes Netz (Vielfaches von 32)")
    ap.add_argument("--height", type=int, default=256, help="Eingabehöhe eigenes Netz (Vielfaches von 32)")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    gpu = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (ohne Grafikkarte, langsam)"
    print(f"Gerät: {device}{gpu}", flush=True)

    paths = ([args.checkpoint] if args.checkpoint else []) + args.checkpoints
    nets = []
    for p in paths:
        if Path(p).exists():
            nets.append((NAMES.get(Path(p).stem, Path(p).stem), load_own(p, device)))
            print(f"Geladen: {p} -> Taste {len(nets)}", flush=True)
        else:
            print(f"Übersprungen, Datei fehlt: {p}", flush=True)
    if not nets:
        raise SystemExit("Kein eigenes Netz gefunden. Beispiel: python live_demo.py --checkpoints rhd.pt --da2")
    da2 = None
    if args.da2:
        try:
            da2 = load_da2(args.da2_model, device, args.offline)
            print("Geladen: Depth Anything V2 Small -> Taste d", flush=True)
        except Exception as e:  # z. B. offline und noch nie heruntergeladen
            print(f"Depth Anything nicht geladen ({type(e).__name__}): {e}\n"
                  "Einmal mit Internet und ohne --offline starten, dann ist es gespeichert.", flush=True)
    da2_w, da2_h = (int(v) for v in args.da2_size.lower().split("x"))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Keine Kamera gefunden. Mit --camera 1 eine andere probieren.")
    win = "GameDepth Live-Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    full = args.fullscreen
    if full:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    active, show_da2 = 0, da2 is not None
    color_own, color_da2 = Colorizer(), Colorizer()
    ms_own = ms_da2 = fps = 0.0
    last = time.perf_counter()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Kamera liefert kein Bild mehr.", flush=True)
                break
            frame = crop_to_panel(cv2.flip(frame, 1))  # wie ein Spiegel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t = time.perf_counter()
            x = to_tensor(cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_AREA)).to(device)
            near = np.exp(-nets[active][1](x)[0, 0].float().cpu().numpy())  # 1/Tiefe: groß = nah
            ms_own += 0.1 * ((time.perf_counter() - t) * 1000 - ms_own)
            panels = [cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA), color_own(near)]
            labels = ["Kamera", f"GameDepth: {nets[active][0]} · {ms_own:.0f} ms"]

            if show_da2:
                t = time.perf_counter()
                x = to_tensor(cv2.resize(rgb, (da2_w, da2_h), interpolation=cv2.INTER_AREA)).to(device)
                disp = da2(pixel_values=x).predicted_depth[0].float().cpu().numpy()  # groß = nah
                ms_da2 += 0.1 * ((time.perf_counter() - t) * 1000 - ms_da2)
                panels.append(color_da2(disp))
                labels.append(f"Depth Anything V2 Small · {ms_da2:.0f} ms")

            now = time.perf_counter()
            fps += 0.1 * (1 / max(now - last, 1e-6) - fps)
            last = now
            width = PANEL_W * len(panels)
            view = np.vstack([title(width, fps), header(width, labels), np.hstack(panels), legend(width)])
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or not window_open(win):
                break
            if ord("1") <= key < ord("1") + len(nets):
                active = key - ord("1")
                color_own = Colorizer()
            elif key in (ord("d"), ord("D")) and da2 is not None:
                show_da2 = not show_da2
            elif key in (ord("f"), ord("F")):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            elif key in (ord("s"), ord("S")):
                name = f"demo_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, view)
                print(f"Gespeichert: {name}", flush=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
