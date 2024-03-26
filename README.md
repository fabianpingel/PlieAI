# 🩰 Plié AI -  Ballet Pose Learning and Improvement with AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plieai.streamlit.app/)

## Installation
1. Clone this repo:
````
git clone https://github.com/fabianpingel/PlieAI.git
````


### Windows

2. Change directory: `cd PlieAI`

3. Install virtual environment:  Double click on `install_venv.bat`

4. Activate virtual environment:  Double click on `start_venv.bat`

5. Install dependencies:
```
pip install -r requirements.txt
```

### Ubuntu

2. Check current python version (`python -V`) and install python3-venv package using
```
sudo apt install python3.8-venv
```
3. Change directory: `cd PlieAI`

4. Install virtual environment:  `./install_venv.sh`

5. Install dependencies:

```
source ./PlieAI_venv/bin/activate
```

```
pip install -r requirements.txt
```


## App

Start app:
```
streamlit run app.py
```

or double click on: `start_streamlit.bat` (if your venv is not yet activated)

## Code

Pose Estimation Code based on [mediapipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python)

