"""Live-Demo für die KI-CON: Webcam-Bild -> Tiefenkarte in Echtzeit.

  python live_demo.py --checkpoint last.pt

Links das Kamerabild, rechts die geschätzte Tiefe (hell = nah). Halte die Hand
vor die Kamera: Sie sollte heller erscheinen als dein Körper.
Tasten: q oder Esc = beenden (oder Fenster schließen), s = Screenshot speichern, f = Vollbild an/aus.
"""
import argparse
import time

import cv2
import numpy as np
import torch

from gamedepth.data import IMAGENET_MEAN, IMAGENET_STD
from gamedepth.model import DepthUNet


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def window_open(win):
    """False, sobald das Fenster mit dem X geschlossen wurde (sonst würde imshow es neu öffnen)."""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=320, help="Netz-Eingabebreite (Vielfaches von 32)")
    ap.add_argument("--height", type=int, default=256, help="Netz-Eingabehöhe (Vielfaches von 32)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = DepthUNet(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    gpu = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (ohne Grafikkarte, langsam)"
    print(f"Gerät: {device}{gpu}", flush=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Keine Kamera gefunden. Mit --camera 1 eine andere probieren.")
    win = "GameDepth: Kamera | geschaetzte Tiefe (hell = nah)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fps, full = 0.0, False
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            frame = cv2.flip(frame, 1)  # wie ein Spiegel
            small = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            log_depth = model(to_tensor(rgb).to(device))[0, 0].float().cpu().numpy()
            inv = np.exp(-log_depth)  # 1/Tiefe: groß = nah
            lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
            v = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
            depth_img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            fps = 0.9 * fps + 0.1 / max(time.time() - t, 1e-6)
            view = np.hstack([small, depth_img])
            view = cv2.resize(view, (view.shape[1] * 2, view.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(view, f"{fps:4.1f} Bilder/s", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or not window_open(win):
                break
            if key == ord("s"):
                cv2.imwrite(f"screenshot_{int(time.time())}.png", view)
            if key == ord("f"):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
