import cv2
import numpy as np
from PIL import Image
import pickle
import os
np.set_printoptions(threshold='nan')

with open("eigen.pickle", "rb") as f:
		tup = pickle.load(f)
with open("data.pickle", "rb") as f:
		names = pickle.load(f)
mean = tup['mean']
phi = tup['phi']
omega = tup['omega']
ids = tup['ids']
count = omega.shape[1]
id_to_names = {names[i] : i for i in names.keys() }
# print(id_to_names)
##Loads the pre-trained classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = raw_input('Press Enter when ready ')
path = 'test/'

cam = cv2.VideoCapture(0)

##Gives the user the oppurtunity to start giving the data after pressing the key 's'
while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if(len(faces) == 0):
		continue
	(x, y, w , h) = faces[0]
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

	cv2.imshow("Press s to give data samples", img)

	if(cv2.waitKey(1) & 0xFF == ord('s')):
		cam.release()
		cv2.destroyAllWindows()
		break
##Starts recording data for the dataset


cam = cv2.VideoCapture(0)

while True:
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if(len(faces) == 0):
		continue
	for (x,y,w,h) in faces:
		# (x, y, w , h) = faces[0]
		image =  gray[y:y+h,x:x+w]
		cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
		image = cv2.resize(image, (128, 128))
		m = image.flatten()
		m = m/np.std(m)
		md = m - mean
		ohm = np.matmul(phi, md)

	##print(w)
		dist = np.zeros(omega.shape)
		for i in range (count):
			dist[:,i] = omega[:,i] - ohm

		dist = dist**2
		dist = np.sum(dist, axis = 0)
		dist = dist**0.5
		##print(dist)
		mx = 10000000000000
		ind = -1
		for i in range(len(dist)):
			if(dist[i] < mx):
				mx = dist[i]
				ind = i
		# print(id_to_names)
		# print()
		# print(ind)
		text  = (id_to_names[ids[ind]])
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (x,y+h)
		fontScale              = 0.7
		fontColor              = (0,255,0)
		lineType               = 2

		cv2.putText(img, text, \
		    bottomLeftCornerOfText, \
		    font, \
		    fontScale, \
		    fontColor,\
		    lineType)\

	cv2.imshow("Data", img)
	# cv2.waitKey(100)
	if((cv2.waitKey(1) & 0xFF) == ord('q')):
		break
cam.release()
cv2.destroyAllWindows()