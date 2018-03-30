import numpy as np
import cv2
import os
path = 'Data'
imagePath = [os.path.join(path, f) for f in os.listdir(path)]

for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	for ssup in range(2):
		nameOfImage = '/home/radhika/Project/FaceRecognition/' + face
		path = os.path.split(face)[-1]
		image = cv2.imread(nameOfImage, 0)

		#cv2.resize(image, (128, 128))
		#image = imutils.resize(image, width=500)
		#image = imutils.resize(image, width=500)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# image = np.array(image)
		# print(image.shape)
		if(ssup == 1):
			image = image[:,::-1]
		cv2.imwrite("AugmentedDataset/"+str(ssup)+path, image)
		