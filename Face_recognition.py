from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mtcnn
import os
from os import listdir
from PIL import Image
from numpy import asarray,savez_compressed
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

from keras.models import load_model
# load the facenet model
model = load_model('facenet_keras.h5')
print('Loaded Model')

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


# load dataset
data = load('nick+ibah2.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print(trainy)
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model_final = SVC(kernel='linear')
model_final.fit(trainX, trainy)

def extract_img(array, required_size=(160, 160)):
    # load image from file
	# image = Image.fromarray(array)
	# # convert to RGB, if needed
	# image = image.convert('RGB')
	# # convert to array
	# pixels = asarray(image)
	# # create the detector, using default weights
	# detector = MTCNN()
	# # detect faces in the image
	# results = detector.detect_faces(pixels)
	# # extract the bounding box from the first face
	# x1, y1, width, height = results[0]['box']
	# # bug fix
	# x1, y1 = abs(x1), abs(y1)
	# x2, y2 = x1 + width, y1 + height
	# # extract the face
	# face = pixels[y1:y2, x1:x2]
	# # resize pixels to the model size
	image = Image.fromarray(array)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

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
    result = detector.detect_faces(img)

    print(len(result))


    for i in range(len(result)):
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']
        
        frame = img_pil[bounding_box[1]:bounding_box[1]+bounding_box[3] + 1,bounding_box[0]:bounding_box[0]+bounding_box[2] + 1]
        
        # plt.imshow(frame)
        # plt.show()
        imgs = asarray(extract_img(frame))
        emb = get_embedding(model, imgs)
        emb = asarray([emb])
        lab = model_final.predict(emb)
        cv2.putText(img, out_encoder.inverse_transform(lab)[0],(bounding_box[0], bounding_box[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
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
