@echo off
rem GameDepth Live-Demo fuer die KI-CON: Doppelklick startet alles.
rem Netze als rhd.pt, mix.pt und gta.pt in diesen Ordner legen (fehlende werden uebersprungen).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python live_demo.py --checkpoints rhd.pt mix.pt gta.pt --da2 --offline --fullscreen
if errorlevel 1 pause
