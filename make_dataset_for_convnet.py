import numpy as np
#import cv2 
import os
import glob
from PIL import Image

'''images_1 = [cv2.imread(files) for files in glob.glob("/project\ data\ images/positive\ cattle\ images/*.bmp")]

print type(images_1)
#print images_1[0]
print len(images_1)
for item in images_1:
    cv2.imshow('image',item)
    print item.shape
    print 'loop enters here'

cv2.waitKey(0)
cv2.destroyAllWindows()'''

#=============ABOVE METHOD DIDNT WORK =================================================

'''mypath='/path/to/folder'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )'''

#===============THESE METHODS ARE TO IMPORT THE IMAGE FILES IN A BULK===================

#path = os.path.join(os.path.dirname(__file__),'/project_data_images/positive_cattle_images/')

'''total = 0

for root,dirs,files in os.walk(path):
    total+=len(files)

for imgs in path:
    print type(imgs)
    cv2.imshow('image',imgs)'''
    
path_pos = './Dataset/extracts/'
save_path1 = './DataSet/Pos_Imgs/'

path_neg = './project_data_images/neg_images/'
save_path2 = './DataSet/Neg_Imgs/'
    
size = (200,200) 

for infile in glob.glob(path_pos + "*.bmp"):
    file__, ext = os.path.splitext(infile)
    file_ = file__[len(path_pos):]
    print file_
    im = Image.open(infile)
    im_= im.resize((200,200))
    print im_.size
    im_.convert('RGB').save(save_path1+file_ +'.jpg', "JPEG")

for infile in glob.glob(path_neg + "*.bmp"):
    file__, ext = os.path.splitext(infile)
    file_ = file__[len(path_neg):]
    print file_
    im = Image.open(infile)
    im_ = im.resize(size)
    #print im_.size
    #im_.save(save_path2+file_[:-4]+'.jpg' , "JPEG")
    im_.convert('RGB').save(save_path2+file_+'.jpg' , "JPEG")
    print im_.size

    
    
#cv2.waitKey(0)
#cv2.destroyAllWindows()

