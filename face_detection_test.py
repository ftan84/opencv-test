"""Just a quick script to show functionality of opencv."""
import numpy as np
import cv2

# ==============================================
# Load up the classifier using frontal face data
# ==============================================
frontalface_location = './opencvdata/haarcacade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(frontalface_location)
#eye_cascade = cv2.CascadeClassifier('./opencvdata/haarcascade_eye.xml')
# =====================================
# Assign the source and the output file
# =====================================
# Use VideoCapture(0) if you want to use webcam
cap = cv2.VideoCapture('./videos/ttd-lgbt.mp4')
out = cv2.VideoWriter('lgbt2.mp4', cv2.cv.CV_FOURCC('X', '2', '6', '4'), 30, (1280, 720))
# ===========================================
# Main loop
# - Iterate through each frame of video
# - Detect face
# - And draw a rectangle around detected face
# ===========================================
counter = 0
while True:
    counter += 1
    ret, frame = cap.read()
    # Change the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply classifier to find face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw a rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    # Draw the frame with the rectangle back
    out.write(frame)
    # cv2.imshow('frame', frame)
    # if counter % 3 == 0:
    #     cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    if ret != True:
        break
# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
