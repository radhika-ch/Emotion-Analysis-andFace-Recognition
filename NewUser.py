import cv2
import pickle
import os

##Finds out if the user already exists.
def findNames(name):
	for i in names.keys():
		if(i == name):
			return 1


##Gets the previously loaded data if any
try:
	with open("data.pickle", "rb") as f:
		names = pickle.load(f)

except:
	names = dict()


##Loads the pre-trained classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = raw_input('Enter your full name: ')
##Checks if user already exists
x = findNames(name)
if(x == 1):
	print('User already exists')
	exit()
names[name] = len(names)+1
path = 'dataset/'

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

for i in range (0, 5):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if(len(faces) == 0):
		i -= 1
		continue
	(x, y, w , h) = faces[0]
	cv2.imwrite("dataset/" + str(names[name]) + "." + str(i) + ".jpg", gray[y:y+h,x:x+w])
	cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
	i = i + 1
	cv2.waitKey(100)
	cv2.imshow("Data", img)
print("Data recorded. Thank you!")
cam.release()
cv2.destroyAllWindows()


##Dumps all the data into a pickle
with open("data.pickle","wb+") as f:
	pickle.dump(names,f)
