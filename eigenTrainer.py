import cv2
import numpy as np
from PIL import Image
import pickle
import os

np.set_printoptions(threshold='nan')
##img.resize((5,25))
path = 'dataset/'
with open("data.pickle", "rb") as f:
		names = pickle.load(f)
imagePath = [os.path.join(path, f) for f in os.listdir(path)]

faces = []
mean = np.zeros((128*128,))
##print(mean.shape)
count = 0
ids = []
for face in imagePath:
	ids.append(int(os.path.split(face)[-1].split('.')[0]))
	image = Image.open(face)
	
	n = np.array(image, 'uint8')
	n = cv2.resize(n, (128, 128))
	m = n.flatten()
	m = m/np.std(m)
	mean = np.add(mean, m)
	faces.append(m)
	##print(m)
	count = count + 1;
mean = np.divide(mean, count)
A = []
for face in faces:
	face = np.subtract(face, mean)
	# print(face)
	A.append(face)

A = np.array(A);
# print(A.shape)
AT = A
A = np.transpose(A)
##print(AT)
cov = np.matmul(AT, A)
# print(cov)
##print(cov.shape[1])
w, v = np.linalg.eig(cov)
tups = []
for i in range(len(w)):
	tups.append((w[i],v[i]))
tups.sort(reverse = True)
k = count-3

psi = []

for i in range (k):
	psi.append(tups[i][1])
psi = np.array(psi)
psi = np.transpose(psi) #equivalnt to u.

phi = np.matmul(np.transpose(psi),AT)
# print(phi.shape)

for i in range(k):
	cv2.imwrite("Eigenfaces/"+str(i)+".png", np.reshape(phi[i],(128,128))*255)

omega = np.matmul(phi, A)
# print(omega.shape)

d = dict()

d['omega'] = omega
d['phi'] = phi
d['mean'] = mean
d['ids'] = ids
with open("eigen.pickle","wb+") as f:
	pickle.dump(d,f)

