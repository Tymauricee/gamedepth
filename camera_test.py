"""Wie viele Bilder pro Sekunde liefert die Webcam? Probiert mehrere Einstellungen durch.

  python camera_test.py

Die Live-Demo kann nicht schneller sein als die Kamera. Viele Laptop-Kameras liefern bei hoher
Auflösung im unkomprimierten Format (YUY2) oder bei wenig Licht (lange Belichtung) nur 10–15
Bilder/s. Dieses Programm misst für jede Kombination aus Kamera-Schnittstelle (Windows: Media
Foundation oder DirectShow), Format (Standard oder MJPG) und Auflösung, wie viele Bilder pro
Sekunde wirklich ankommen. Die beste Zeile überträgt man dann auf live_demo.py.

  python camera_test.py --belichtung

prüft zusätzlich feste Belichtungszeiten (DirectShow, 640 × 480).
"""
import argparse
import sys
import time

import cv2

BACKENDS = ([("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)] if sys.platform == "win32"
            else [("auto", cv2.CAP_ANY)])


def measure(camera, backend, mjpg, size, frames, exposure=None):
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        return None
    if exposure is not None:  # feste Belichtungszeit 2^exposure Sekunden statt Automatik
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10):  # Kamera einschwingen lassen (Belichtung, Weißabgleich)
        ok, frame = cap.read()
    t = time.perf_counter()
    got, brightness = 0, 0.0
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        got += 1
        brightness = float(frame.mean())
    secs = time.perf_counter() - t
    if exposure is not None:  # Automatik wiederherstellen, manche Treiber merken sich die Einstellung
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.release()
    if not got:
        return None
    return got / secs, frame.shape[1], frame.shape[0], brightness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=60, help="Bilder pro Messung")
    ap.add_argument("--belichtung", action="store_true",
                    help="statt aller Kombinationen: feste Belichtungszeiten mit DirectShow, 640x480")
    args = ap.parse_args()
    if args.belichtung:
        test_exposure(args)
        return
    print("Bitte normal vor die Kamera setzen, gutes Licht. Dauert etwa eine halbe Minute.\n", flush=True)
    print(f"{'Schnittstelle':14s}{'Format':10s}{'gewünscht':>12s}{'geliefert':>12s}{'Bilder/s':>10s}{'Helligkeit':>12s}")
    for name, backend in BACKENDS:
        for mjpg in (False, True):
            for size in ((640, 480), (1280, 720)):
                r = measure(args.camera, backend, mjpg, size, args.frames)
                fmt = "MJPG" if mjpg else "Standard"
                if r is None:
                    print(f"{name:14s}{fmt:10s}{f'{size[0]}x{size[1]}':>12s}{'–':>12s}{'geht nicht':>10s}", flush=True)
                    continue
                fps, w, h, light = r
                print(f"{name:14s}{fmt:10s}{f'{size[0]}x{size[1]}':>12s}{f'{w}x{h}':>12s}{fps:10.1f}{light:12.0f}",
                      flush=True)
    print("\nHelligkeit: 0 = schwarz, 255 = weiß. Unter etwa 60 verlängern viele Kameras die Belichtung"
          " und liefern dann weniger Bilder/s – mehr Licht hilft.")


def test_exposure(args):
    """Bei automatischer Belichtung wartet die Kamera bei wenig Licht länger pro Bild. Eine feste,
    kurze Belichtung erzwingt mehr Bilder/s, macht das Bild aber dunkler."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    print(f"{'Belichtung':22s}{'Bilder/s':>10s}{'Helligkeit':>12s}")
    for exp in (None, -4, -5, -6, -7):
        r = measure(args.camera, backend, False, (640, 480), args.frames, exp)
        name = "automatisch" if exp is None else f"{exp} (1/{2 ** -exp} s)"
        if r is None:
            print(f"{name:22s}{'geht nicht':>10s}", flush=True)
            continue
        fps, _, _, light = r
        print(f"{name:22s}{fps:10.1f}{light:12.0f}", flush=True)
    print("\nGesucht: die Zeile mit etwa 30 Bilder/s, deren Helligkeit noch über etwa 60 liegt.")


if __name__ == "__main__":
    main()
