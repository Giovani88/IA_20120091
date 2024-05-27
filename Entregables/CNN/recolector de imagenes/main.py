import numpy as np
import cv2 as cv
import math 
import os
 
directorio = "robocasa"
cap = cv.VideoCapture("../assets/videos/robo casa/robocasa f.mp4")
i=0
#contador = 0
contador=len(os.listdir(directorio))
#print(i)
frame_interval = 1  # Capturar una imagen cada 30 frames (ajusta este valor según necesites)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #cv.imshow('img', frame)
    if i % frame_interval == 0:
        resized_frame = cv.resize(frame, (28, 21))
        cv.imshow('img', resized_frame)
        cv.imwrite(f'{directorio}/data{contador}.jpg', resized_frame)
        contador+=1
        
    i=i+1    
    #frame = cv.resize(frame, (28, 21))
    #cv.imshow('img', frame)
    #if k == ord('a'):
    
    #cv.imwrite('tornados/data'+str(i)+'.jpg', frame )
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()