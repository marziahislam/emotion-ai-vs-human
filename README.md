# AI vs. Human Facial Emotion Judgments

Compares how LibreFace and human participants judge ambiguous fear–happy morphed faces
from Sun et al. (2023).

## Setup
### DeepFace
```bash
conda create -n deepface_env python=3.10 -y
conda activate deepface_env
pip install -r requirements-deepface.txt
python scripts/run_deepface.py
```
Output: `results/deepface_probs.csv`, with 7 emotion probabilities per image plus `p_fear_2afc`.


### LibreFace (in progress)
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