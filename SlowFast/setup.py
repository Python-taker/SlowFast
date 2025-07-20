#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="slowfast",
    version="1.0",
    author="FAIR",
    url="unknown",
    description="SlowFast Video Understanding",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "av",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "detectron2",
        "opencv-python",
        "pandas",
        "torchvision>=0.4.2",
        "Pillow",  # OK
        "scikit-learn",  # fixed from "sklearn"
        "tensorboard",
        # "fairscale",  # optional, can be commented if errors
    ],
    extras_require={"tensorboard_video_visualization": ["moviepy"]},
    packages=find_packages(exclude=("configs", "tests")),
)
