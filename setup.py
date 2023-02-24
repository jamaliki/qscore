#!/usr/bin/env python

"""
Setup module for Q-score
"""

import os
import sys

from setuptools import setup

sys.path.insert(0, f"{os.path.dirname(__file__)}/qscore")

import qscore

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)), "qscore")

setup(
    name="qscore",
    entry_points={
        "console_scripts": [
            "qscore = qscore.__main__:main",
        ],
    },
    version=qscore.__version__,
)
