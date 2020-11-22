"""
Implementation  of various machine learning models
nym 2020
"""

import argparse
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

class image_classifier():
    """ Class defining image properties """
    def __init__(self, width, height, n_channels):
        self.im_width  = width
        self.im_height = height
        self.im_channels = n_channels

    def training_prep(self, data_location):
        """ Training preparation """
        dog_or_cat = []
        tr_files = os.listdir(data_location)
        for t in tr_files:
            if t.split('.')[0] == 'dog':
                dog_or_cat.append(1)
            else:
                dog_or_cat.append(0)
        self.df = pd.DataFrame({
            'filename':tr_files,
            'category':dog_or_cat
         })
    
    def training(self):
        """ Model training """
        pass