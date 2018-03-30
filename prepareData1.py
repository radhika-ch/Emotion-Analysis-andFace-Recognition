import cv2
import os
import numpy as np
from PIL import Image

path = 'data1'
imagePath = [os.path.join(path, f) for f in os.listdir(path)]
i=1
for face in imagePath:
	#image = cv2.imread("/home/radhika/facial-landmarks/images/2.2.jpg")
	image = cv2.imread(face, 0)
	
	x = os.path.split(face)[-1].split('.')[0].split('_')[-1]	
	print(str(i) + '.' + x + ".jpg")
	image = np.array(image)
	image = cv2.resize(image, (128, 128))
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	cv2.imwrite("Data/" + str(i) + '_' + x + ".jpg", image)
	i = i+1