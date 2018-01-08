import numpy as np
#import cv2 
import os
import glob
from PIL import Image

pos_filelist  = glob.glob('./DataSet/Pos_Imgs/*.jpeg')
print(len(pos_filelist))
X_pos = np.array([np.array(Image.open(fname)) for fname in pos_filelist])
n_pos = X_pos.shape[0]
np.random.shuffle(X_pos)


#Setting the labels for positive images
Y_pos = []
for i in range(n_pos):
  Y_pos.append(1)

  
Y_pos = np.array(Y_pos)
print("N_pos", n_pos)


#getting Negative Images  
neg_filelist  = glob.glob('./DataSet/Neg_Imgs/*.jpg')
print(len(neg_filelist))
X_neg = np.array([np.array(Image.open(fname)) for fname in neg_filelist])
n_neg = X_neg.shape[0]
np.random.shuffle(X_neg)

#Settting the labels for negative images
Y_neg = []
for i in range(n_neg):
  Y_neg.append(0)
  
Y_neg = np.array(Y_neg)


print("N_neg = ", n_neg)


print ("x_NEG SHAPE:", X_neg.shape)
  
X_train_pos = X_pos[int(0.7*n_pos):]
X_test_pos = X_pos[-int(0.3*n_pos):]
Y_train_pos = Y_pos[int(0.7*n_pos):]
Y_test_pos = Y_pos[-int(0.3*n_pos):]
  
X_train_neg = X_neg[int(0.7*n_neg):]
X_test_neg = X_neg[-int(0.3*n_neg):]
Y_train_neg = Y_neg[int(0.7*n_neg):]
Y_test_neg = Y_neg[-int(0.3*n_neg):]

print ("x_TRAIN-POS SHAPE:", X_train_pos.shape)
print ("x_TRAIN-NEG SHAPE:", X_train_neg.shape)
  
train_data = np.concatenate((X_train_pos,X_train_neg),axis=0)
#train_data = X_trin_pos.dstack(X_train_neg)
train_labels_ = np.concatenate((Y_train_pos,Y_train_neg),axis=0)
#train_labels = Y_train_pos.dstack

eval_data = np.concatenate((X_test_pos,X_test_neg),axis=0)

for i in range(train_data.shape[0]):
    if i <= X_train_pos.shape[0]:
        img = Image.fromarray(train_data[i])
        img.save('./Data/Train/Pos/'+str(i)+'.jpg','JPEG')
    else:
        img = Image.fromarray(train_data[i])
        img.save('./Data/Train/Neg/'+str(i)+'.jpg','JPEG')
        
        
for i in range(eval_data.shape[0]):
    if i <= X_test_pos.shape[0]:
        img = Image.fromarray(eval_data[i])
        img.save('./Data/Test/Pos/'+str(i)+'.jpg','JPEG')
    else:
        img = Image.fromarray(eval_data[i])
        img.save('./Data/Test/Neg/'+str(i)+'.jpg','JPEG')
        
