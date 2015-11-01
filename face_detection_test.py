import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./opencvdata/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('./opencvdata/haarcascade_eye.xml')

#img = cv2.imread('ftan.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture('./videos/ttd-lgbt.mp4')
out = cv2.VideoWriter('lgbt2.mp4', cv2.cv.CV_FOURCC('X', '2', '6', '4'), 30, (1280, 720))

counter = 0
while(True):
    counter += 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    out.write(frame)

    # cv2.imshow('frame', frame)
    # if counter % 3 == 0:
    #     cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    if ret != True:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
