import cv2
import numpy



def frame_reshape(img):
    print (img.shape)
    print (type(img))
    
    img_feed = cv2.resize(img,(200,200))
    
    print(img_feed.shape)
    #print(img_feed)
    cv2.imshow('frame2',img_feed)
    
    return img_feed
    
#cv2.destroyAllWindows()

def video_extract():
    cap = cv2.VideoCapture('testVid.avi')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        print (ret, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame2 = frame_reshape(frame)
        print ("Flag-1")
        cv2.imshow('frame',gray)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
video_extract()
