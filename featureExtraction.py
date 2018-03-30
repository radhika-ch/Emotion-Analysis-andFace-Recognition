# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
import imutils
from imutils import face_utils 
import numpy as np
import argparse
from PIL import Image
import dlib
import cv2
import os
import pickle
from collections import OrderedDict

np.set_printoptions(threshold='nan')
feature_set = []
labels = ['0', '1', '2', '3', '4']
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
	#help="/home/radhika/facial-landmarks/shape_predictor_68_face_landmarks.dat")
#ap.add_argument("-i", "--image", required=True,
#	help="/home/radhika/facial-landmarks/images/2.2.jpg")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
d = dict()
# load the input image, resize it, and convert it to grayscale
path = 'AugmentedDataset'
imagePath = [os.path.join(path, f) for f in os.listdir(path)]

for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	
	nameOfImage = '/home/radhika/Project/FaceRecognition/' + face
	path = os.path.split(face)[-1]
	image = cv2.imread(nameOfImage, 0)

	#cv2.resize(image, (128, 128))
	#image = imutils.resize(image, width=500)
	#image = imutils.resize(image, width=500)
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = np.array(image)
	# detect faces in the grayscale image
	rects = detector(image, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		#(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		if(shape.shape != 0):
			for (x, y) in shape:
				cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
			cv2.imwrite("Face_landmarks/"+ path , image)
	
	vectors = []
	for i in range (17, 67):
		vectors.append(shape[i]-shape[27])
	vectors = np.array(vectors)
	label = np.zeros(5);
	label[labels.index(os.path.split(nameOfImage)[-1].split('.')[0].split('_')[-1])] = 1;
	vectors = vectors.flatten()
	feature = [vectors,label]
	feature = np.array(feature)
	feature_set.append(feature)
	#print(pose)
	#print(cfg.all_joints_names)
	#print(cfg.all_joints)
np.random.shuffle(feature_set)
with open("features.pickle","wb") as f:
	pickle.dump(feature_set,f)
