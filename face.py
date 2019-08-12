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
    img_pil = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img_pil)

    for i in range(len(result)):
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']
        
        frame = img_pil[bounding_box[0]-50:bounding_box[0]+bounding_box[2] + 51,bounding_box[1] - 50:bounding_box[1]+bounding_box[3] + 51]
        

        imgs = asarray(extract_img(frame))
        emb = get_embedding(model, imgs)
        emb = asarray([emb])
        lab = model_final.predict(emb)
        cv2.putText(img, out_encoder.inverse_transform(lab)[0],(bounding_box[0], bounding_box[1]),4,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(img,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,155,255),
                        2)

#         cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
#         cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
#         cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
#         cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
#         cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

