from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import pandas
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

pos_filelist  = glob.glob('./DataSet/Pos_Imgs/*.jpg')

for i in pos_filelist:
    img = Image.open(i) #load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='tmp', save_prefix='file', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
