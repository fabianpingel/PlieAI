#!/bin/bash

# Hier den Namen der virtuellen Umgebung angeben
VENV="PlieAI_venv"

echo "1. Virtuelle Umgebung erzeugen"
python3 -m venv $VENV

echo "2. Virtuelle Umgebung aktivieren"
source ./$VENV/bin/activate

echo "3. ipykernel installieren"
pip install ipykernel

echo "4. PIP Upgrade"
python -m pip install --upgrade pip

echo "5. ipykernel in Umgebung installieren"
python -m ipykernel install --name=$VENV

echo "6. Install Jupyter Lab"
pip install jupyterlab

echo "7. Start Jupyter lab"
jupyter lab


