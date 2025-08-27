# MotionAIS
An AI-based toolbox for tracking and analyzing the 3D motion of the back surface in patients with AIS.


# Getting started

## Cloning

Begin by cloning the repositery in the directory of your choice using the following commands:

```shell
git clone https://github.com/VisionICLab/MotionAIS.git
cd MotionAIS
```

## Creating environment
Now create the python environment and download all dependencies to make the interface work.
Be careful as the interface was made with python 3.9.19, you can try other versions but libraries compatibility was only tested on this one.

### Creating environment using venv

```shell
{/path/to/python3.9} -m venv .venv
source .venv/bin/activate
```

### Upgrading pip (just in case)

```shell
python -m pip install --upgrade pip
```

###  Installing dependencies

the interface uses the following libraries:

- segmentation models pytorch
- opencv
- open3d
- Kivy
- Kivy garden (matplotlib)
- skimage

```shell
python -m pip install -r requirements.txt
```
