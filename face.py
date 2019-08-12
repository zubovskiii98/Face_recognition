import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haar.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
detector = MTCNN()


while 1:
    ret, img = cap.read()
    result = detector.detect_faces(img)

    for i in range(len(result)):
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']

        cv2.rectangle(img,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,155,255),
                        2)

        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
        
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

