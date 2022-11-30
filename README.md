# 3d_motion_generator

## Task Description
Given a list of motions and a desired motion speed, the tool strings the motions together and converts them to plausible motion with specified speed through our model.


# Getting started
## Installation
This code was tested with Pytoch 1.12.1, CUDA 11.8, Python 3.8.13 and Ubuntu 20.04.5

- Clone this repo:

```
git clone https://github.com/cwl0429/3d_motion_generator.git
```

- Change dir

```
cd 3d_motion_generator
```

- Create model folder

```
mkdri model
```

- Download models

[model](https://drive.google.com/file/d/1iXDmnqapE7vur89ivaM4VFc69JwIWd2r/view?usp=share_link)

- Create environment

```
conda env create -f environment.yml
```

- Run tool

```
python main.py -i input_path -o output_path  
```
