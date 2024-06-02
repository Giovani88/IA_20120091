import cv2 as cv
import os
 
directorio = "robocasa"
cap = cv.VideoCapture("../assets/videos/robo casa/robocasa f.mp4")
i=0
contador=len(os.listdir(directorio))
frame_interval = 1  # Capturar una imagen cada n frames

while True:
    ret, frame = cap.read()
    if not ret:
        break    
    if i % frame_interval == 0:
        resized_frame = cv.resize(frame, (28, 21))
        cv.imshow('img', resized_frame)
        cv.imwrite(f'{directorio}/data{contador}.jpg', resized_frame)
        contador+=1
        
    i=i+1    
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()