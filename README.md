# AI vs. Human Facial Emotion Judgments

Compares how LibreFace and human participants judge ambiguous fear–happy morphed faces
from Sun et al. (2023).

## Setup
```bash
conda create -n libreface_env python=3.9 -y
conda activate libreface_env
conda install -c conda-forge "cmake<4" dlib=19.24.6 -y
pip install -r requirements.txt
```

## Data
From https://osf.io/26rhz/: copy `01 Stimuli/` images into `data/stimuli/` and
`behaviorData.mat` into `data/`.

## Run
```bash
python scripts/run_libreface.py --limit 3  
python scripts/run_libreface.py
```
Outputs go to `results/`: `libreface_probs.csv` (emotion scores per image, plus
`p_fear_2afc`) and `libreface_features.npy` (512-d features per image, for RSA).