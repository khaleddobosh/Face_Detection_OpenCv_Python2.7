# Face_Detection_OpenCv_Python2.7
#Done By Mohamad Nazzal 
import numpy as np
import cv2
cascPath = "C:\Users\m7mdn\Desktop\haar\haarcascade_frontalface_default.xml"
eye_cascade = "C:\Users\m7mdn\Desktop\haar\haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyecascade = cv2.CascadeClassifier(eye_cascade)

image = cv2.imread("C:/Users/m7mdn/Desktop/facehd.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)
eyes = eyecascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)

    )
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
  
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

print("Found {0} faces!".format(len(faces)))
print("Found {0} eyes!".format(len(eyes)))

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
