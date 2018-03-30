import cv2
import os
import numpy as np
from PIL import Image

path1 = 'Neutral'
path2 = 'Happy'
path3 = 'Sad'
path4 = 'Angry'
path5 = 'Surprised'

imagePath = [os.path.join(path1, f) for f in os.listdir(path1)]
i=1
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)

	image = np.array(image)
	#image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + "_0.jpg", image)
	i = i+1

imagePath = [os.path.join(path2, f) for f in os.listdir(path2)]
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)

	image = np.array(image)
	#image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + "_1.jpg", image)
	i = i+1

imagePath = [os.path.join(path3, f) for f in os.listdir(path3)]
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)

	image = np.array(image)
	#image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + "_2.jpg", image)
	i = i+1

imagePath = [os.path.join(path4, f) for f in os.listdir(path4)]
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)

	image = np.array(image)
	#image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + "_3.jpg", image)
	i = i+1

imagePath = [os.path.join(path5, f) for f in os.listdir(path5)]
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)

	image = np.array(image)
	#image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + "_4.jpeg", image)
	i = i+1