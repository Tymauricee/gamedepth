"""Test mit echten Webcam-Fotos: Erkennt ein Modell, ob die Hand vor oder hinter dem Körper ist?

Alle anderen Ergebnisse stammen von gerenderten Bildern. Dieses Programm prüft dieselbe
Frage (Handtest, Trennschärfe) an echten Fotos und beantwortet damit, ob das Netz den
Sprung vom Spiel in die Wirklichkeit schafft.

Schritt 1 – Fotos aufnehmen (am Laptop mit Webcam):

  python webcam_test.py aufnehmen --ordner webcam_fotos

  Klick ins Bild  Countdown (3 s), danach wird ein Foto gemacht (Leertaste geht auch)
  Maus ziehen     Rahmen eng um die HAND, danach Rahmen um ein Stück OBERKÖRPER (ohne Hand);
                  ein Rahmen gilt, sobald die Maustaste losgelassen wird
  VORNE / HINTEN  anklicken (oder V / H): Hand ist näher an / weiter weg von der Kamera als der
                  Oberkörper; „verwerfen“ (oder X) löscht das Foto
  q oder X oben   beenden (bei erneutem Start werden neue Fotos angehängt)

  Ziel: etwa gleich viele Fotos „vorne“ und „hinten“ (je 15–20), und in beiden Fällen
  die Hand mal höher, mal tiefer als den Oberkörper halten. Nur dann lässt sich prüfen,
  ob ein Modell nach der Bildhöhe rät („tiefer im Bild = näher“).

Schritt 2 – Auswerten:

  python webcam_test.py auswerten --ordner webcam_fotos --checkpoints rhd.pt mix.pt --da2

Für jedes Modell: mittlere Tiefe (Median) in der Mitte des Hand-Rahmens und in der Mitte
des Oberkörper-Rahmens (ohne den Teil, der im Hand-Rahmen liegt) -> sagt das Modell
„Hand vorne“? Gemessen wird nur in den mittleren 60 % jedes Rahmens, weil die Ränder
auch Hintergrund enthalten. Daraus
Trefferquoten und Trennschärfe = (Treffer bei „vorne“ + Treffer bei „hinten“) / 2.
Raten ergibt 0,5. Der 95-%-Bereich (Bootstrap) zeigt, wie unsicher die Zahl bei
wenigen Fotos ist.

Schreibt ergebnis.json und webcam_beispiele.jpg in den Fotoordner. Mit --zeige fehler stattdessen
webcam_fehler.jpg: nur die Fotos, auf denen mindestens ein Modell falsch liegt.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from gamedepth.data import IMAGENET_MEAN, IMAGENET_STD

LABELS = "labels.json"
MIN_PIXELS = 20  # so viele Netz-Pixel muss jeder Rahmen mindestens haben
INNER = 0.6      # gemessen wird nur in den mittleren 60 % jedes Rahmens
GREEN, BLUE, WHITE, BLACK = (0, 200, 0), (255, 120, 0), (255, 255, 255), (0, 0, 0)


# ---------------------------------------------------------------- Hilfen
def load_labels(folder):
    path = Path(folder) / LABELS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_labels(folder, labels):
    (Path(folder) / LABELS).write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")


def text(img, s, y, scale=0.6, color=WHITE):
    """Schrift auf abgedunkeltem Hintergrund, damit sie überall lesbar ist (nur ASCII)."""
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y0, y1, x1 = max(y - th - 8, 0), min(y + base + 4, img.shape[0]), min(tw + 20, img.shape[1])
    img[y0:y1, 0:x1] = (img[y0:y1, 0:x1] * 0.3).astype(np.uint8)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def label_bar(img, s, color=WHITE):
    """Beschriftung auf einem schwarzen Balken am unteren Bildrand."""
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 24), (img.shape[1], h), BLACK, -1)
    cv2.putText(img, s, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img


def draw_boxes(img, hand, body=None):
    for box, color, name in ((hand, GREEN, "Hand"), (body, BLUE, "Koerper")):
        if box is None:
            continue
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, name, (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def is_lower(rec):
    """Liegt die Mitte des Hand-Rahmens tiefer im Bild als die des Oberkörper-Rahmens?"""
    return rec["hand"][1] + rec["hand"][3] / 2 > rec["koerper"][1] + rec["koerper"][3] / 2


def counts(labels):
    front = sum(r["hand_vorne"] for r in labels)
    return front, len(labels) - front


# ---------------------------------------------------------------- Schritt 1: Aufnehmen
DISPLAY_WIDTH = 960  # Fensterbreite in Pixeln, unabhängig von der Kameraauflösung
STEPS = {
    "hand": "Schritt 1/3: Maustaste gedrueckt halten und Rahmen eng um die HAND ziehen",
    "body": "Schritt 2/3: Rahmen um ein Stueck OBERKOERPER ziehen (ohne Hand)",
    "label": "Schritt 3/3: Ist die Hand naeher an der Kamera als der Oberkoerper?",
}
BUTTONS = [("v", "VORNE (V)", GREEN), ("h", "HINTEN (H)", BLUE), ("x", "verwerfen (X)", (90, 90, 90))]


class Mouse:
    """Merkt sich Klicks und gezogene Rahmen im Kamerafenster (in Fensterkoordinaten)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.click = self.start = self.now = self.box = None

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            (x0, y0), self.start = self.start, None
            if abs(x - x0) < 8 and abs(y - y0) < 8:
                self.click = (x, y)
            else:
                self.box = [min(x0, x), min(y0, y), abs(x - x0), abs(y - y0)]


def window_open(win):
    """False, sobald das Fenster mit dem X geschlossen wurde (sonst würde imshow es neu öffnen)."""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def read_key(delay):
    """Taste als Kleinbuchstabe ('' = keine). Groß/klein egal, damit die Feststelltaste nicht stört."""
    key = cv2.waitKey(delay)
    return "" if key == -1 else chr(key & 0xFF).lower()


def button_rects(view):
    h, w = view.shape[:2]
    bw, bh, gap = 230, 54, 20
    x0 = (w - len(BUTTONS) * bw - (len(BUTTONS) - 1) * gap) // 2
    return [(x0 + i * (bw + gap), h - bh - 20, bw, bh) for i in range(len(BUTTONS))]


def draw_buttons(view):
    for (key, label, color), (x, y, w, h) in zip(BUTTONS, button_rects(view)):
        cv2.rectangle(view, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(view, (x, y), (x + w, y + h), WHITE, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(view, label, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2,
                    cv2.LINE_AA)


def button_at(view, point):
    if point is None:
        return ""
    for (key, _, _), (x, y, w, h) in zip(BUTTONS, button_rects(view)):
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            return key
    return ""


def next_name(folder, labels):
    used = {r["datei"] for r in labels}
    i = len(labels) + 1
    while f"foto_{i:03d}.jpg" in used or (folder / f"foto_{i:03d}.jpg").exists():
        i += 1
    return f"foto_{i:03d}.jpg"


def record(args):
    folder = Path(args.ordner)
    folder.mkdir(parents=True, exist_ok=True)
    labels = load_labels(folder)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Keine Kamera gefunden. Läuft noch die Live-Demo? Sonst mit --camera 1 probieren.")
    win = "GameDepth Webcam-Test"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)  # feste Größe: Mauskoordinaten = Bildpunkte
    mouse = Mouse()
    cv2.setMouseCallback(win, mouse)
    print("Kamerafenster ist offen. Klick INS KAMERABILD, um ein Foto auszuloesen (3 s Countdown).\n"
          "Beenden: q im Kamerafenster oder das Fenster mit dem X schliessen.", flush=True)

    state, countdown_end, photo, hand, body = "live", None, None, None, None
    while True:
        if state in ("live", "countdown"):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Kamera liefert kein Bild. Läuft noch die Live-Demo oder ein anderes Kameraprogramm?")
            frame = cv2.flip(frame, 1)  # wie ein Spiegel, wie in der Live-Demo
            src = frame
        else:
            src = photo
        scale = DISPLAY_WIDTH / src.shape[1]
        view = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        to_view = lambda b: [int(v * scale) for v in b] if b else None

        if state == "live":
            front, back = counts(labels)
            text(view, f"Fotos: {len(labels)}  (vorne {front}, hinten {back})", 30, 0.7)
            text(view, "Klick ins Bild = Foto (3 s Countdown)    q = Ende", 60, 0.7)
        elif state == "countdown":
            left = countdown_end - time.time()
            cv2.putText(view, str(int(left) + 1), (view.shape[1] // 2 - 40, view.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, WHITE, 8, cv2.LINE_AA)
        else:
            draw_boxes(view, to_view(hand), to_view(body))
            text(view, STEPS[state], 30, 0.7, (0, 255, 255))
            if state == "label":
                text(view, "Rahmen neu: R", 60, 0.6)
                draw_buttons(view)
            else:
                text(view, "Rahmen neu: R     Foto verwerfen: X", 60, 0.6)
                if mouse.start is not None:  # Rahmen, der gerade gezogen wird
                    cv2.rectangle(view, mouse.start, mouse.now, GREEN if state == "hand" else BLUE, 2)
        cv2.imshow(win, view)
        key = read_key(15)
        if key == "q" or not window_open(win):
            break

        if state == "live":
            if key == " " or mouse.click:
                state, countdown_end = "countdown", time.time() + args.countdown
                print("Countdown laeuft ...", flush=True)
            mouse.click = mouse.box = None  # nicht reset(): ein Klick kann gerade erst begonnen haben
        elif state == "countdown":
            if time.time() >= countdown_end:
                state, photo, hand, body = "hand", frame.copy(), None, None
                mouse.reset()
                print("Foto gemacht. Jetzt im Fenster mit gedrueckter Maustaste einen Rahmen um die Hand ziehen.",
                      flush=True)
        elif state in ("hand", "body") or (state == "label" and key == "r"):
            if key == "r":
                state, hand, body = "hand", None, None
                print("Rahmen neu: zuerst wieder die Hand.", flush=True)
            elif key in ("x", "\x1b"):
                state = "live"
                print("Verworfen.", flush=True)
            elif mouse.box and mouse.box[2] >= 10 and mouse.box[3] >= 10:
                box = [int(round(v / scale)) for v in mouse.box]
                if state == "hand":
                    hand, state = box, "body"
                    print("Hand-Rahmen gespeichert. Jetzt Rahmen um ein Stueck Oberkoerper ziehen.", flush=True)
                else:
                    body, state = box, "label"
                    print("Oberkoerper-Rahmen gespeichert. Jetzt VORNE oder HINTEN anklicken (oder V / H).",
                          flush=True)
            mouse.box = mouse.click = None
        elif state == "label":
            choice = key if key in ("v", "h", "x") else ("x" if key == "\x1b" else button_at(view, mouse.click))
            mouse.click = mouse.box = None
            if choice == "x":
                state = "live"
                print("Verworfen.", flush=True)
            elif choice in ("v", "h"):
                name = next_name(folder, labels)
                cv2.imwrite(str(folder / name), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec = {"datei": name, "hand": hand, "koerper": body, "hand_vorne": choice == "v",
                       "breite": int(photo.shape[1]), "hoehe": int(photo.shape[0])}
                labels.append(rec)
                save_labels(folder, labels)
                state = "live"
                print(f"Gespeichert: {name} | Hand {'vorne' if rec['hand_vorne'] else 'hinten'}"
                      f" | Hand {'tiefer' if is_lower(rec) else 'hoeher'} als Oberkoerper", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    front, back = counts(labels)
    lower = sum(is_lower(r) for r in labels)
    print(f"\n{len(labels)} Fotos in {folder}: Hand vorne {front}, hinten {back}; "
          f"Hand tiefer als Oberkoerper {lower}, hoeher {len(labels) - lower}.")


# ---------------------------------------------------------------- Schritt 2: Auswerten
def model_name(path):
    p = Path(path)
    return p.parent.name if p.stem in ("last", "best") else p.stem


def nearness_predictors(args, device):
    """Jedes Modell liefert eine Karte „Nähe“: groß = nah (für Vergleich und Bild)."""
    from analyze import da2_predictor, net_predictor
    preds = {}
    for path in args.checkpoints:
        if not Path(path).exists():
            print(f"Übersprungen, Datei fehlt: {path}")
            continue
        depth = net_predictor(path, device)  # liefert Tiefe
        preds[model_name(path)] = lambda img, f=depth: 1.0 / np.maximum(f(img), 1e-6)
    if args.da2:
        preds["depth_anything_v2_small"] = da2_predictor(device, args.da2_model)  # liefert schon inverse Tiefe
    if not preds:
        raise SystemExit("Kein Modell angegeben: --checkpoints und/oder --da2")
    return preds


def to_input(bgr, size):
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def box_masks(rec, size, inner=INNER):
    """Rahmen auf Netz-Auflösung umrechnen und nur ihren mittleren Teil nehmen (inner = Anteil
    von Breite und Höhe): Ein Rahmen enthält an den Rändern auch Hintergrund, die Mitte fast
    nur Hand bzw. Oberkörper. Vom Oberkörper wird zusätzlich der ganze Hand-Rahmen abgezogen."""
    w, h = size
    sx, sy = w / rec["breite"], h / rec["hoehe"]
    masks = []
    for x, y, bw, bh in (rec["hand"], rec["koerper"]):
        cx, cy, hw, hh = (x + bw / 2) * sx, (y + bh / 2) * sy, bw * sx / 2, bh * sy / 2
        for full in (True, False):
            f = 1.0 if full else inner
            m = np.zeros((h, w), bool)
            m[max(int(cy - hh * f), 0):int(np.ceil(cy + hh * f)), max(int(cx - hw * f), 0):int(np.ceil(cx + hw * f))] = True
            masks.append(m)
    hand_full, hand, _, body = masks
    return hand, body & ~hand_full


def bootstrap(gt, pred, reps=2000, seed=0):
    """95-%-Bereich der Trennschärfe durch Ziehen mit Zurücklegen."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(gt), len(gt))
        g, p = gt[i], pred[i]
        if g.all() or not g.any():
            continue
        vals.append((p[g].mean() + (~p[~g]).mean()) / 2)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def summarize(gt, pred, lower, min_n):
    def share(sel):  # Anteil „Hand vorne“ unter den ausgewählten Fotos
        return float(pred[sel].mean()) if sel.sum() >= min_n else float("nan")

    def mean(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else float("nan")

    hit_front = float(pred[gt].mean()) if gt.any() else float("nan")
    hit_back = float((~pred[~gt]).mean()) if (~gt).any() else float("nan")
    groups = {f"hand_{'tiefer' if lo else 'hoeher'}_{'vorne' if g else 'hinten'}": (lo, g)
              for lo in (True, False) for g in (True, False)}
    return {
        "n": int(len(gt)),
        "hand_acc": float((pred == gt).mean()),
        "treffer_hand_vorne": hit_front,
        "treffer_hand_hinten": hit_back,
        "trennschaerfe": (hit_front + hit_back) / 2,
        "trennschaerfe_95": bootstrap(gt, pred),
        "sagt_hand_vorne": float(pred.mean()),
        "einfluss_wahrheit": mean([share((lower == lo) & gt) - share((lower == lo) & ~gt) for lo in (True, False)]),
        "einfluss_hoehe": mean([share(lower & (gt == g)) - share(~lower & (gt == g)) for g in (True, False)]),
        "gruppen": {k: {"n": int(((lower == lo) & (gt == g)).sum()),
                        "sagt_hand_vorne": share((lower == lo) & (gt == g))} for k, (lo, g) in groups.items()},
    }


def mcnemar(right_a, right_b):
    """Gepaarter Vergleich zweier Modelle auf denselben Fotos. Zählt nur Fotos, auf denen sich
    die beiden uneinig sind: b = nur A richtig, c = nur B richtig. Wären beide gleich gut, wäre
    jedes dieser Fotos ein Münzwurf; p = Wahrscheinlichkeit für eine mindestens so einseitige
    Verteilung (exakter Binomialtest, zweiseitig)."""
    from math import comb
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p


def colorize(near):
    lo, hi = np.percentile(near, 2), np.percentile(near, 98)  # wie in der Live-Demo
    v = np.clip((near - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def pick_examples(labels, k):
    """Abwechselnd „vorne“ und „hinten“, damit beide Fälle im Bild vorkommen."""
    front = [i for i, r in enumerate(labels) if r["hand_vorne"]]
    back = [i for i, r in enumerate(labels) if not r["hand_vorne"]]
    out = []
    while len(out) < k and (front or back):
        for lst in (front, back):
            if lst and len(out) < k:
                out.append(lst.pop(0))
    return out


def example_sheet(images, labels, names, maps, verdicts, idx, size):
    w, h = size
    head = np.full((30, w * (1 + len(names)), 3), 255, np.uint8)
    for j, title in enumerate(["Foto mit Rahmen"] + names):
        cv2.putText(head, title[:28], (j * w + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    rows = [head]
    for i in idx:
        rec = labels[i]
        sx, sy = w / rec["breite"], h / rec["hoehe"]
        scale = lambda b: [int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)]
        photo = draw_boxes(cv2.resize(images[i], size, interpolation=cv2.INTER_AREA),
                           scale(rec["hand"]), scale(rec["koerper"]))
        label_bar(photo, "Wahrheit: Hand " + ("vorne" if rec["hand_vorne"] else "hinten"))
        tiles = [photo]
        for n in names:
            tile = colorize(maps[n][i])
            ok = verdicts[n][i] == rec["hand_vorne"]
            label_bar(tile, f"sagt {'vorne' if verdicts[n][i] else 'hinten'}: {'richtig' if ok else 'falsch'}",
                      (120, 255, 120) if ok else (120, 120, 255))
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def evaluate(args):
    folder = Path(args.ordner)
    labels = load_labels(folder)
    if not labels:
        raise SystemExit(f"Keine Fotos in {folder} ({LABELS} fehlt oder ist leer). Erst 'aufnehmen' ausführen.")
    w, h = (int(v) for v in args.size.lower().split("x"))
    if w % 32 or h % 32:
        raise SystemExit("--size muss aus Vielfachen von 32 bestehen, z. B. 320x256")
    size = (w, h)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"Gerät: {device}{gpu} | Fotos: {len(labels)} | Netz-Eingabe {w}x{h}", flush=True)

    images, keep = [], []
    for rec in labels:
        img = cv2.imread(str(folder / rec["datei"]))
        hand, body = box_masks(rec, size)
        if img is None or hand.sum() < MIN_PIXELS or body.sum() < MIN_PIXELS:
            print(f"Übersprungen: {rec['datei']} (Datei fehlt oder Rahmen zu klein)")
            continue
        images.append(img)
        keep.append(rec)
    labels = keep
    gt = np.array([r["hand_vorne"] for r in labels])
    lower = np.array([is_lower(r) for r in labels])

    preds = nearness_predictors(args, device)
    names = list(preds)
    maps, verdicts, results = {}, {}, {}
    with torch.no_grad():
        for name, predict in preds.items():
            maps[name], verdicts[name] = [], []
            for img, rec in zip(images, labels):
                near = predict(to_input(img, size))[0]
                hand, body = box_masks(rec, size)
                maps[name].append(near)
                verdicts[name].append(bool(np.median(near[hand]) > np.median(near[body])))
            results[name] = summarize(gt, np.array(verdicts[name]), lower, args.min_gruppe)
            print(f"{name}: fertig", flush=True)
    results["hoehenregel"] = summarize(gt, lower.copy(), lower, args.min_gruppe)  # „tiefer im Bild = näher“
    right = {n: np.array(verdicts[n]) == gt for n in names}
    pairs = []
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            b, c, pval = mcnemar(right[a], right[b_name])
            pairs.append({"a": a, "b": b_name, "nur_a_richtig": b, "nur_b_richtig": c, "p": pval})
    results["_paarvergleich"] = pairs
    results["_fotos"] = [{"datei": r["datei"], "hand_vorne": bool(g), "hand_tiefer": bool(lo),
                          "sagt_hand_vorne": {n: bool(verdicts[n][k]) for n in names}}
                         for k, (r, g, lo) in enumerate(zip(labels, gt, lower))]

    clean = json.loads(json.dumps(results).replace("NaN", "null"))
    (folder / "ergebnis.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.zeige == "fehler":  # nur Fotos, auf denen mindestens ein Modell falsch liegt
        idx = [i for i, r in enumerate(labels) if any(verdicts[n][i] != r["hand_vorne"] for n in names)]
        sheet_name = "webcam_fehler.jpg"
        if not idx:
            print("Kein Modell liegt auf irgendeinem Foto falsch – kein Fehlerbild.")
    else:
        idx = pick_examples(labels, args.beispiele)
        sheet_name = "webcam_beispiele.jpg"
    if idx:
        sheet = example_sheet(images, labels, names, maps, verdicts, idx[:args.beispiele], size)
        cv2.imwrite(str(folder / sheet_name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    front = int(gt.sum())
    print(f"\n{len(gt)} Fotos: Hand vorne {front}, hinten {len(gt) - front}. "
          f"Hand tiefer als Oberkörper: {int(lower.sum())}.")
    print("\n" + "modell".ljust(26) + "Treffer vorne  Treffer hinten  Trennschärfe  (95-%-Bereich)  Einfluss Wahrheit  Einfluss Höhe")
    models = {n: r for n, r in results.items() if not n.startswith("_")}
    for name, r in models.items():
        lo, hi = r["trennschaerfe_95"]
        print(f"{name:26s}{r['treffer_hand_vorne']:13.0%}{r['treffer_hand_hinten']:16.0%}{r['trennschaerfe']:14.2f}"
              f"     ({lo:.2f}–{hi:.2f}){r['einfluss_wahrheit']:19.2f}{r['einfluss_hoehe']:15.2f}")
    print(f"\nGruppen (Anzahl Fotos | Anteil „Hand vorne“ je Modell; leer = weniger als {args.min_gruppe} Fotos):")
    for g, c in results[names[0]]["gruppen"].items():
        line = "  ".join(f"{n}: {r['gruppen'][g]['sagt_hand_vorne']:.0%}" for n, r in models.items())
        print(f"  {g:24s} n={c['n']:3d} | {line}")
    if pairs:
        print("\nPaarvergleich auf denselben Fotos (nur Fotos, bei denen sich die beiden uneinig sind):")
        for q in pairs:
            print(f"  {q['a']} gegen {q['b']}: nur {q['a']} richtig {q['nur_a_richtig']}, nur {q['b']} richtig "
                  f"{q['nur_b_richtig']}  ->  p = {q['p']:.3f}" + ("  (Unterschied belegt)" if q["p"] < 0.05 else ""))
    print("\nLesehilfe: Trennschärfe 0,5 = Raten, 1,0 = immer richtig. Bei wenigen Fotos ist der 95-%-Bereich"
          " breit. Ob ein Modell besser ist als ein anderes, zeigt der Paarvergleich: p unter 0,05 = Unterschied"
          " belegt, sonst kann er Zufall sein.")
    print(f"Gespeichert: {folder / 'ergebnis.json'}" + (f" und {folder / sheet_name}" if idx else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="schritt", required=True)
    a = sub.add_parser("aufnehmen", help="Fotos mit der Webcam aufnehmen und beschriften")
    a.add_argument("--ordner", default="webcam_fotos")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--countdown", type=float, default=3.0, help="Sekunden bis zum Foto")
    e = sub.add_parser("auswerten", help="Modelle auf den Fotos vergleichen")
    e.add_argument("--ordner", default="webcam_fotos")
    e.add_argument("--checkpoints", nargs="*", default=[])
    e.add_argument("--da2", action="store_true", help="Depth Anything V2 Small mit auswerten")
    e.add_argument("--da2-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    e.add_argument("--size", default="320x256", help="Netz-Eingabe Breite x Höhe, wie in der Live-Demo")
    e.add_argument("--beispiele", type=int, default=6, help="Fotos im Beispielbild")
    e.add_argument("--zeige", choices=["gemischt", "fehler"], default="gemischt",
                   help="Beispielbild: gemischt (vorne/hinten abwechselnd) oder nur Fotos mit Fehlern -> webcam_fehler.jpg")
    e.add_argument("--min-gruppe", type=int, default=3, help="Mindestzahl Fotos je Gruppe für Anteile")
    args = ap.parse_args()
    record(args) if args.schritt == "aufnehmen" else evaluate(args)


if __name__ == "__main__":
    main()
