# coding: utf-8
"""
Data reading helper functions for the indoor location challenge
"""
import os
import numpy as np

"""
Basic information such as data location and magic numbers / dictionaries
"""
KAGGLE_TRAIN_DIR = os.getenv("KAGGLE_TRAIN_DIR")

REPEAT_SSIDS = set(np.loadtxt("repeat_ssids.txt", dtype=str).tolist())
