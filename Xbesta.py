import cv2
import numpy as np

cap = cv2.VideoCapture(0)
rostodetect= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos= rostodetect.detectMultiScale(cinza, 1.2, 8)
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x,y), (x + w, y+ h), (255, 0, 0), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
