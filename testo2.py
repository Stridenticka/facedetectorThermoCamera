import numpy as np
import cv2 as cv
from matplotlib.pyplot import gray




def FaceRcogntionhaarcascade(fileLocation):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    img = cv.imread(fileLocation)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def FaceRcogntionhaarcascadebefore():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    img = cv.imread('C:/Users/user09/Pictures/camerathermicmen.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def LBPTraining():
    # load cascade classifier training file for lbpcascade
    lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')

    # load test image
    test2 = cv2.imread('data/test2.jpg')

    # call our function to detect faces
    faces_detected_img = detect_faces(lbp_face_cascade, test2)

    # convert image to RGB and show image
    cv.imshow('img', faces_detected_img)



groupe1Location="C:/Users/user09/Pictures/TheRealGroup-400x304.jpg"
groupe2Location="C:/Users/user09/Pictures/equipe.jpg"
thermic1Location="C:/Users/user09/Pictures/camerathermicmen.jpg"
thermic2Location='C:/Users/user09/Pictures/image.jpg'
thermic3Location='C:/Users/user09/Pictures/testVisageThe/3visages1.png'
thermic4Location='C:/Users/user09/Pictures/testVisageThe/3visages2.png'
thermic5Location='C:/Users/user09/Pictures/testVisageThe/testbn.png'



FaceRcogntionhaarcascade(thermic5Location)