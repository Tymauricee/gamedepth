# GameDepth

Ein CNN lernt aus gerenderten 3D-Bildern, wie weit jedes Pixel von der Kamera entfernt ist, und zeigt das live an einer normalen Webcam: Hältst du die Hand vor die Kamera, erscheint sie in der Tiefenkarte heller (näher) als dein Körper.

Maurice Tay · Projekt im Modul *Advances in Intelligent Systems* (Hochschule Düsseldorf, WiSe 2026/27).

## Idee

Eine Webcam misst keine Entfernung, sie sieht nur Farben. Ein neuronales Netz kann Tiefe trotzdem schätzen, wenn es mit vielen Bildern trainiert wurde, deren echte Tiefe bekannt ist. 3D-Engines berechnen diese Tiefe für jedes Bild ohnehin (Tiefenpuffer). GameDepth trainiert deshalb ausschließlich mit gerenderten Daten:

| Datensatz | Inhalt | Rolle |
| --- | --- | --- |
| [RHD](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) | 41.258 Blender-Renderings von Figuren mit Händen nah an der Kamera, 320 × 320, mit Tiefe und Hand-/Körpermaske | Hauptdatensatz, passt zur Demo |
| [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html) | 12.000 Bilder aus GTA V mit Tiefenpuffer | Vergleich: Straßenszenen statt Nahaufnahmen |
| Mischung | 70 % RHD, 30 % GTA pro Epoche | prüft, ob vielfältigere Szenen helfen oder Abkürzungen aus GTA mitbringen |

## Aufbau

```
gamedepth/data.py      Datensätze laden, Augmentierung
gamedepth/model.py     U-Net mit ResNet-18-Encoder (14,4 Mio. Parameter)
gamedepth/losses.py    skaleninvarianter Log-Verlust + Kanten-Term
gamedepth/metrics.py   AbsRel, RMSE, δ1 und Handtest
train.py               Training
evaluate.py            Auswertung eines Netzes mit Beispielbildern
analyze.py             Vergleich mit Depth Anything V2 und Test „unten im Bild = näher“
check_data.py          Datensatz vor dem Training prüfen
live_demo.py           Webcam-Demo
webcam_test.py         Test mit echten Webcam-Fotos (Hand vorne/hinten)
notebooks/gamedepth_colab.ipynb   alles Schritt für Schritt auf Google Colab
```

## Training auf Google Colab

1. `gamedepth.zip` nach Google Drive in `MyDrive/gamedepth/` hochladen.
2. `notebooks/gamedepth_colab.ipynb` in Colab öffnen, Laufzeit auf *L4 GPU* stellen.
3. Zellen der Reihe nach ausführen.

Lokal geht es genauso: `pip install -r requirements.txt`, dann

```bash
python check_data.py --data rhd --root data/rhd
python train.py --data rhd --root data/rhd --pretrained --aug --epochs 15 --out runs/rhd_pre_aug
python train.py --data mix --root data/rhd --gta-root data/gta --gta-share 0.3 --pretrained --aug --out runs/mix30_pre_aug
python evaluate.py --rhd data/rhd --checkpoint runs/rhd_pre_aug/last.pt
python analyze.py --rhd data/rhd --da2 --checkpoints runs/rhd_pre_aug/last.pt runs/gta_pre_aug/last.pt
```

## Auf dem eigenen Laptop (Windows mit NVIDIA-Grafikkarte)

Die Schritte sind für ein ASUS TUF Gaming A14 mit RTX 4050 (6 GB) geschrieben. PyTorch rechnet nur auf der NVIDIA-Karte; die eingebaute Radeon-Grafik wird nicht genutzt.

1. Python 3.12 von python.org installieren, im Installer *Add python.exe to PATH* ankreuzen.
2. Diesen Ordner entpacken, z. B. nach `C:\Users\<name>\gamedepth`. Im Explorer in die Adresszeile `cmd` tippen und Enter drücken: Es öffnet sich eine Eingabeaufforderung in diesem Ordner.
3. Umgebung anlegen und PyTorch **mit CUDA** installieren. Die Umgebung liegt bewusst unter einem kurzen Pfad (`C:\venvs`): Windows erlaubt nur 260 Zeichen pro Dateipfad, und PyTorch und transformers enthalten Dateien mit sehr langen Namen; in einem tief verschachtelten Projektordner bricht die Installation sonst ab. Den PyTorch-Befehl gibt pytorch.org unter *Get Started* aus (Stable, Windows, Pip, Python, neueste CUDA-Version):

   ```bat
   python -m venv C:\venvs\gamedepth
   C:\venvs\gamedepth\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
   pip install -r requirements.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   Die letzte Zeile muss `True NVIDIA GeForce RTX 4050 Laptop GPU` ausgeben. Bei `False`: NVIDIA-Treiber über die NVIDIA App aktualisieren oder auf pytorch.org eine ältere CUDA-Version wählen. Wichtig: erst PyTorch mit CUDA, dann `requirements.txt`, sonst landet die Version ohne GPU-Unterstützung auf dem Rechner.
4. Die trainierten Netze aus Google Drive herunterladen (`MyDrive/gamedepth/runs/<lauf>/last.pt`) und in den Ordner legen, z. B. als `rhd.pt` und `mix.pt`.
5. Bei jedem neuen Öffnen der Eingabeaufforderung zuerst `C:\venvs\gamedepth\Scripts\activate`.

Die Umgebung liegt absichtlich unter einem kurzen Pfad: Windows erlaubt ohne Zusatzeinstellung nur 260 Zeichen pro Dateipfad, und PyTorch enthält tief verschachtelte Dateien. In einem langen Projektordner bricht die Installation sonst ab.

Für flüssige Bilder Netzteil anschließen und in Armoury Crate den Modus *Turbo* oder *Leistung* wählen; im Akkubetrieb drosselt die Grafikkarte.

## Live-Demo

```bash
python live_demo.py --checkpoint rhd.pt
```

Links das Kamerabild, rechts die geschätzte Tiefe (hell = nah). Oben links steht, wie viele Bilder pro Sekunde das Netz schafft. Tasten: `q` beenden, `s` Screenshot, `f` Vollbild. Eine externe USB-Webcam wählt man mit `--camera 1`.

## Test mit echten Fotos

Alle anderen Zahlen stammen von gerenderten Bildern. `webcam_test.py` prüft dieselbe Frage an echten Webcam-Fotos:

```bash
python webcam_test.py aufnehmen --ordner webcam_fotos
python webcam_test.py auswerten --ordner webcam_fotos --checkpoints rhd.pt mix.pt --da2
```

Beim Aufnehmen löst ein Klick ins Kamerabild einen Countdown von 3 s aus. Danach zieht man mit gedrückter Maustaste einen engen Rahmen um die Hand und einen um ein Stück Oberkörper und klickt auf *VORNE* (Hand näher an der Kamera als der Oberkörper) oder *HINTEN* (Hand weiter weg). `R` zieht die Rahmen neu, `X` verwirft das Foto, `q` oder das Schließen des Fensters beendet. Gemessen wird der Median der geschätzten Tiefe in der Mitte beider Rahmen. Die Auswertung gibt Trefferquoten, Trennschärfe mit 95-%-Bereich (Bootstrap), den Vergleich mit der Höhenregel und ein Beispielbild aus. Die Fotos zeigen den Aufnehmenden und gehören deshalb nicht ins Repository (`.gitignore`).

## Auswertung

| Maß | Bedeutung |
| --- | --- |
| AbsRel, RMSE, δ1 | Standardmaße der Tiefenschätzung nach Median-Skalierung |
| Handtest | Anteil der Bilder, in denen das Netz die Reihenfolge Hand/Körper richtig schätzt |
| Basisrate | Ergebnis eines Netzes, das immer „Hand vorne“ sagt; der Handtest muss deutlich darüber liegen |
| Verhältnisfehler | Abweichung des Verhältnisses Handtiefe/Körpertiefe, strenger als die Reihenfolge |
| Einfluss Wahrheit / Höhe | ob die Antwort „Hand vorne“ von der echten Tiefe abhängt oder nur davon, ob die Hand im Bild tiefer liegt (`analyze.py`) |

## Ergebnisse (Kurzfassung)

**Gerenderte Testbilder** (RHD, 2.728 Bilder):

| Modell | Tiefe der Person, AbsRel (kleiner = besser) | Trennschärfe Hand vorne/hinten (0,5 = Raten) |
| --- | --- | --- |
| eigenes Netz, trainiert auf RHD | 0,114 | 0,83 |
| eigenes Netz, 70 % RHD + 30 % GTA | 0,119 | 0,80 |
| eigenes Netz, trainiert auf GTA | 0,164 | 0,51 |
| Depth Anything V2 Small (zum Vergleich) | 0,132 | 0,90 |
| flache Schätzung (überall gleiche Tiefe) | 0,176 | 0,50 |

**Echte Webcam-Fotos** (41 Fotos, eine Person, ein Raum; Hand vor der Brust oder flach an der Wand hinter der Person):

| Modell | Hand vorne erkannt | Hand hinten erkannt | Trennschärfe (95-%-Bereich) |
| --- | --- | --- | --- |
| eigenes Netz, RHD | 16/16 | 17/25 | 0,84 (0,74–0,93) |
| eigenes Netz, RHD + GTA | 16/16 | 14/25 | 0,78 (0,68–0,88) |
| Depth Anything V2 Small | 16/16 | 25/25 | 1,00 |

Live-Demo: 55 Bilder/s bei 320 × 256 auf einer RTX 4050 (Laptop).

**Was die Netze aus den Spieldaten zusätzlich gelernt haben:** Das GTA-Netz schätzt nach der Bildhöhe („weiter unten = näher“). Das RHD-Netz hat gelernt, dass die Hand meist vorne ist und dass alles außer der Person weit weg ist (in RHD ist der Hintergrund immer ein flaches Foto in 5 m Entfernung); auf echten Fotos erscheint der Raum deshalb schwarz. Liegt die Hand hinter dem Körper auf Hüfthöhe, zeigen die eigenen Netze sie meist so nah wie die Hüfte. Alle Zahlen stammen aus je einem Trainingslauf; zwei Läufe mit gleichen Einstellungen unterschieden sich im AbsRel des ganzen Bildes um bis zu 0,025.

## Lizenzen

Code: MIT (siehe `LICENSE`). RHD: nur Forschungszwecke, keine kommerzielle Nutzung. MVS-Synth: nur Forschung und Lehre. Depth Anything V2 Small: Apache 2.0. Die Datensätze und die trainierten Netze sind nicht Teil dieses Repositorys; die Datensätze gibt es über die Links oben.

## Hinweis zur KI-Nutzung

Teile des Codes und der Dokumentation wurden mit Unterstützung von Claude (Anthropic) erstellt und vom Autor geprüft.
