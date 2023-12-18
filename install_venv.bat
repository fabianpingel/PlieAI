@echo off

REM ### Hier den Namen der virtuellen Umgebung angeben ###
set VENV=PlieAI_venv

echo 1. Virtuelle Umgebung erzeugen
@echo on
python -m venv %VENV%
@echo off

echo 2. Virtuelle Umgebung aktivieren
@echo on
call .\%VENV%\Scripts\activate
@echo off

echo 3. ipykernel installieren
@echo on
pip install ipykernel
@echo off

echo 4. PIP Upgrade
@echo on
python.exe -m pip install --upgrade pip
@echo off

echo 5.ipykernel in Umgebung installieren
@echo on
python -m ipykernel install --name=%VENV%
@echo off

echo 6.Install Jupyter Lab
@echo on
pip install jupyterlab
@echo off

echo 7. Start Jupyter lab
@echo on
jupyter lab
@echo off

pause