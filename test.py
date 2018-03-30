import cv2
import numpy as np
from PIL import Image
import pickle
import os

np.set_printoptions(threshold='nan')

image = Image.open("test.png")
n = np.array(image, 'uint8')
m = n.flatten()
print(m.shape)

