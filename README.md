# clef-2024-authority

## installing the environment

(install miniconda, checking the box too add it to PATH: https://docs.anaconda.com/free/miniconda/) and restart the command prompt

```
conda create -n clef python=3.8
conda activate clef
```

then

```
pip install -e .
```

## adding the clef GitLab repo to conveniently load data

at the repo root (here) execute 

```
git clone https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.git
```