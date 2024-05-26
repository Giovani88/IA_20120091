import numpy as np
import cv2 as cv
import math 
 
cap = cv.VideoCapture("/home/likcos/Vídeos/rosas.mp4")
i=0
while True:
    ret, frame = cap.read()
    cv.imshow('img', frame)
    k = cv.waitKey(1)
    #if k == ord('a'):
    i=i+1    
    cv.imwrite('/home/likcos/datasetcubre/data'+str(i)+'.jpg', frame )
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()