"""
Utility script to prepare data
nym 2020
"""

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
import numpy as np
import hdfdict

def read_image(file,im_sz):
    """ 
    read image and return dictionary with labels: 1 -> cat, 0 -> dog
    and image matrix 
    """
    img = image.load_img(file, target_size=(im_sz,im_sz),interpolation='bicubic')
    x = image.img_to_array(img)
    if "cat" == file.split('.')[0]:
        label = 1
    else: label = 0
    return x,label

def prepare_data(img_test, where_to_save):
    cur_dir  = os.getcwd()
    os.chdir(img_test)
    files = os.listdir()
    numfiles = len(files)
    im_sz = 136
    im_list  = []
    lbl_list = []
    count    = 0
    for k in files:
        count += 1
        im,lbl = read_image(k,im_sz)
        lbl_list.append(lbl)
        im_list.append(im)
        print(f'Processed {count} / {numfiles}')
    training_data = {'input':im_list,'output':lbl_list}
    os.chdir(cur_dir)
    hdfdict.dump(training_data,where_to_save)

if __name__ == "__main__":
    img_test = "data/train/"
    print("=========== Train data ===============")
    prepare_data(img_test,"datasets/training_cat_dogs.h5")
    print("=========== Test data ===============")
    img_test = "data/test1/test1/"
    prepare_data(img_test,"datasets/testing_cat_dogs.h5")
    


