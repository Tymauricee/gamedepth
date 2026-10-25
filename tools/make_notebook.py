"""Erzeugt notebooks/gamedepth_colab.ipynb (Google Colab)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# GameDepth auf Google Colab

Dieses Notebook trainiert das Tiefen-CNN auf einer Colab-GPU.

**Vorher einmalig:** `gamedepth.zip` in deinem Google Drive in den Ordner `MyDrive/gamedepth/` hochladen.
Dann oben im Menü *Laufzeit → Laufzeittyp ändern → L4 GPU* wählen und die Zellen der Reihe nach ausführen.
Lass den Browser-Tab offen, solange das Training läuft. Nach dem Training: *Laufzeit → Verbindung trennen und Laufzeit löschen*, sonst verbraucht die GPU weiter Recheneinheiten.
Checkpoints und Ergebnisse landen in deinem Drive und bleiben erhalten, wenn Colab die Sitzung beendet.
""")

md("## 1. GPU prüfen und Google Drive einbinden")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
PROJEKT = '/content/drive/MyDrive/gamedepth'
""")

md("## 2. Code entpacken und Pakete installieren\nPyTorch, torchvision und OpenCV sind in Colab schon installiert.")
code("""
!mkdir -p {PROJEKT}/runs
!rm -rf /content/gamedepth && unzip -q {PROJEKT}/gamedepth.zip -d /content
%cd /content/gamedepth
!pip install -q OpenEXR transformers
""")

md("""
## 3. RHD herunterladen (7,1 GB)

Beim ersten Mal wird das ZIP in dein Drive geladen, danach nur noch lokal entpackt (das geht schneller als direkt aus Drive zu lesen).
Falls der Download fehlschlägt: auf der [Datensatzseite](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) mit Rechtsklick auf *RHD_v1-1.zip* den Link kopieren und unten bei `RHD_URL` eintragen.
Der Datensatz ist nur für Forschungszwecke freigegeben.
""")
code("""
import os
RHD_URL = 'https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip'
zip_path = f'{PROJEKT}/RHD_v1-1.zip'
if not os.path.exists(zip_path):
    !wget -q --show-progress -O "{zip_path}" "{RHD_URL}"
!mkdir -p /content/data/rhd && unzip -q -n "{zip_path}" -d /content/data/rhd
!ls /content/data/rhd
""")

md("""
## 4. Daten prüfen

Wichtig vor dem ersten Training: Die Tiefenwerte sollten zwischen 0 und 5 m liegen, die Hand sollte meist näher sein als die Person.
Falls eine **WARNUNG** erscheint, stimmt die Tiefen-Kodierung nicht, dann bitte nicht trainieren.
""")
code("""
!python check_data.py --data rhd --root /content/data/rhd --out /content/datencheck.jpg
from IPython.display import Image, display
display(Image('/content/datencheck.jpg', width=640))
""")

md("""
## 5. Erstes Training: RHD, vortrainierter Encoder, Augmentierung

Jede Epoche wird auf dem RHD-Testteil ausgewertet. Achte auf **Handtest** und **Basisrate**: Der Handtest ist nur dann gut, wenn er deutlich über der Basisrate liegt.
Für einen ersten schnellen Test kannst du `--epochs 2 --limit 2000` anhängen.
""")
code("""
!python train.py --data rhd --root /content/data/rhd --pretrained --aug --epochs 15 --out {PROJEKT}/runs/rhd_pre_aug
""")

md("## 6. Auswertung und Beispielbilder\nJede Zeile: Foto | echte Tiefe | Schätzung des Netzes (hell = nah).\n\nAusgewertet wird `last.pt` (letzte Epoche), nicht `best.pt`: `best.pt` wurde anhand der Testbilder ausgesucht. Wer mit den Testdaten auswählt, schönt das Ergebnis.")
code("""
!python evaluate.py --rhd /content/data/rhd --checkpoint {PROJEKT}/runs/rhd_pre_aug/last.pt --out {PROJEKT}/eval
display(Image(f'{PROJEKT}/eval/rhd_pre_aug_beispiele.jpg', width=720))
""")

md("""
## 7. Experimente für das Paper

Gleiches Netz, jeweils eine Sache anders. Jeder RHD-Lauf dauert so lange wie Schritt 5 (rund eine Stunde auf der L4), der GTA-Lauf rund 25 Minuten.
**Achtung:** Ein Lauf mit demselben `--out`-Ordner überschreibt den alten. Für einen zweiten Durchgang einen neuen Namen wählen, z. B. `--out {PROJEKT}/runs/rhd_pre_aug_2`.

| Lauf | Trainingsdaten | Vortraining | Augmentierung |
| --- | --- | --- | --- |
| rhd_pre_aug | RHD | ja | ja |
| rhd_pre | RHD | ja | nein |
| rhd_scratch_aug | RHD | nein | ja |
| gta_pre_aug | GTA V (MVS-Synth) | ja | ja |
""")
code("""
!python train.py --data rhd --root /content/data/rhd --pretrained --epochs 15 --out {PROJEKT}/runs/rhd_pre
!python train.py --data rhd --root /content/data/rhd --aug --epochs 15 --out {PROJEKT}/runs/rhd_scratch_aug
""")

md("### GTA V als Vergleich (9 GB, nur für Forschung und Lehre)\n\nBeim ersten Mal wird das Archiv in dein Drive geladen, danach nur noch entpackt.")
code("""
gta_zip = f'{PROJEKT}/GTAV_540.tar.gz'
if not os.path.exists(gta_zip):
    !wget -q --show-progress -O "{gta_zip}" https://huggingface.co/datasets/phuang17/MVS-Synth/resolve/main/GTAV_540.tar.gz
!mkdir -p /content/data/gta && tar -xzf "{gta_zip}" -C /content/data/gta
!python check_data.py --data gta --root /content/data/gta --out /content/datencheck_gta.jpg
!python train.py --data gta --root /content/data/gta --eval-root /content/data/rhd --pretrained --aug --epochs 15 --out {PROJEKT}/runs/gta_pre_aug
""")

md("""
### Gemischt: 70 % RHD (Blender), 30 % GTA

Vorschlag von Prof. Müller. Pro Epoche werden so viele Bilder gezogen wie bei einem reinen RHD-Lauf, im Mittel 30 % davon aus GTA. Gleiche Rechenzeit wie Schritt 5 (rund eine Stunde). Die Zelle entpackt GTA, falls das in dieser Sitzung noch nicht passiert ist.
""")
code("""
import glob
if not glob.glob('/content/data/gta/*'):
    !mkdir -p /content/data/gta && tar -xzf "{PROJEKT}/GTAV_540.tar.gz" -C /content/data/gta
!python train.py --data mix --root /content/data/rhd --gta-root /content/data/gta --gta-share 0.3 --pretrained --aug --epochs 15 --out {PROJEKT}/runs/mix30_pre_aug
""")

md("""
## 8. Vergleich und Fehleranalyse

Wertet das RHD-Netz, das gemischte Netz, das GTA-Netz, eine flache Schätzung und Depth Anything V2 Small (fertiges Großmodell, Bezugswert) auf denselben 2.728 Testbildern aus.
Zusätzlich prüft es die Hypothese „weiter unten im Bild = näher“: `einfluss_wahrheit` nahe 1 und `einfluss_hoehe` nahe 0 heißt, das Modell richtet sich nach der echten Tiefe; umgekehrt richtet es sich nach der Bildhöhe. Dauer: wenige Minuten.
""")
code("""
!python analyze.py --rhd /content/data/rhd --out {PROJEKT}/eval --da2 --baseline --checkpoints {PROJEKT}/runs/rhd_pre_aug/last.pt {PROJEKT}/runs/mix30_pre_aug/last.pt {PROJEKT}/runs/gta_pre_aug/last.pt
""")

md("## 9. Alle Ergebnisse als Tabelle")
code("""
import json, glob, pandas as pd
rows = []
for f in sorted(glob.glob(f'{PROJEKT}/runs/*/log.csv')):
    last = pd.read_csv(f).iloc[-1].to_dict(); last['lauf'] = f.split('/')[-2] + ' (Training, letzte Epoche)'; rows.append(last)
for name, r in json.load(open(f'{PROJEKT}/eval/analyse.json')).items():
    r = {k: v for k, v in r.items() if k != 'gruppen'}; r['lauf'] = name + ' (analyze.py)'; rows.append(r)
cols = ['absrel', 'delta1', 'absrel_person', 'absrel_person_affin', 'delta1_person_affin', 'hand_acc', 'hand_base', 'trennschaerfe', 'hand_ratio_err', 'einfluss_wahrheit', 'einfluss_hoehe', 'hand_n']
pd.DataFrame(rows).set_index('lauf').reindex(columns=cols).round(3)
""")

md("## Fertig für heute?\nLaufzeit beenden, damit keine Recheneinheiten mehr verbraucht werden. Alles Wichtige liegt in deinem Drive.")
code("""
from google.colab import runtime
runtime.unassign()
""")

md("""
## 10. Live-Demo auf deinem Laptop

`last.pt` aus `MyDrive/gamedepth/runs/rhd_pre_aug/` herunterladen. Auf dem Laptop im entpackten `gamedepth`-Ordner:

```
pip install torch torchvision opencv-python
python live_demo.py --checkpoint last.pt
```
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parents[1] / "notebooks" / "gamedepth_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out)
