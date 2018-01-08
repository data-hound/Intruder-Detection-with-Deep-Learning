from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import pandas
import time
import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.metrics import precision_score

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(200,200,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    print ("Flag-01")
    
    return model

#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])

def load_weights(model):
    model.load_weights('first_try.h5')
    
    return model
    
def get_model():
    model_ = create_model()
    model = load_weights(model_)
    
    print ("Flag-02")
    
    return model
    
def get_pred(img, model):
    print ("Flag-10")
    pred = model.predict(img)
    print ("Flag-20")
    img_1 = np.reshape(img, (200,200,3)) 
    img_ = Image.fromarray(img_1,'RGB')
    print ("Flag-30")
    #img_.show()
    print (pred)
    if pred == 1:
        img_.save('./Predictions/Pos/'+str(datetime.datetime.now().time())+'.jpg', "JPEG")
        #img_.show()
    else:
        img_.save('./Predictions/Neg/'+str(datetime.datetime.now().time())+'.jpg', "JPEG")
        #img_.show()
    #time.sleep(3)
    print ("Flag-40")
    

def frame_reshape(img):
    print (img.shape)
    print (type(img))
    
    img_feed_ = cv2.resize(img,(200,200))
    img_feed = np.reshape(img_feed_, (1,200,200,3))
    
    print(img_feed.shape)
    #print(img_feed)
    #cv2.imshow('frame2',img_feed)
    print ("Flag-41")
    return img_feed
    
def video_extract(model):
    cap = cv2.VideoCapture('testVid2.avi')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        print (ret, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame2 = frame_reshape(frame)
        print ("Flag-50")
        get_pred(frame2, model)

        #cv2.imshow('frame',gray)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

conv_net = get_model()
video_extract(conv_net)
