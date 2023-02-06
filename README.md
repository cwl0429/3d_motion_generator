# 3d_motion_generator

## Task Description
Given a list of motions and a desired motion speed, the tool strings the motions together and converts them to plausible motion with specified speed through our model.

## Demo

![](https://i.imgur.com/nNGdxfm.gif)

# Getting started
## Installation
This code was tested with Pytoch 1.12.1, CUDA 11.8, Python 3.8.13 and Ubuntu 20.04.5

- Clone this repo

```
git clone https://github.com/cwl0429/3d_motion_generator.git
```

- Change dir

```
cd 3d_motion_generator
```

- Create model folder

```
mkdir model
```

- Download models

[model link](https://drive.google.com/file/d/1iXDmnqapE7vur89ivaM4VFc69JwIWd2r/view?usp=share_link)

- Unzip models and put it in model

- Intsall pytorch with proper cuda version

```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

- Install modules

```
pip install -r requirements.txt
```

- Run tool

`-i`: input file path <br/>
`-o`: output file path 

```
python main.py -i template.xlsx -o template  
```
