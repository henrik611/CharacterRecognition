import sys
import cv2
import numpy as np

# Training
samples = np.loadtxt('train_data/samples.txt',np.float32)
labels = np.loadtxt('train_data/labels.txt',np.float32)
labels = labels.reshape((labels.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, labels)

# Recognition

DEBUG = False
FILE_NAME = 'test_data/bienso' + sys.argv[1] + '.jpg'
PLATE_SIZE = (190 * 3, 140 * 3)
THRESHOLD_VALUE = 120
MIN_CHAR_WIDTH = 20
MAX_CHAR_WIDTH = 100
MIN_CHAR_HEIGHT = 140
MAX_CHAR_HEIGHT = 200
CHAR_SIZE = (20, 50)

def custom_cmp(img1, img2):
	if (abs(img1[1] - img2[1]) < 100):
		return img1[0] - img2[0]
	else:
		return img1[1] - img2[1]


src = cv2.imread(FILE_NAME)
src = cv2.resize(src, PLATE_SIZE)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 7, 20, 20)
ret, thresh = cv2.threshold(filtered, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY);
# thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY, 11, 2)

thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 1000))
response = []
results = []

if (DEBUG):
	cv2.imshow('image', thresh)
	cv2.waitKey(0)

for contour in contours:
	[x, y, w, h] = cv2.boundingRect(contour)
	area = cv2.contourArea(contour, True)
	if area > 0 and MIN_CHAR_WIDTH < w and w < MAX_CHAR_WIDTH and MIN_CHAR_HEIGHT < h and h < MAX_CHAR_HEIGHT:
		single_char = thresh[y: y+ h, x: x + w]

		if (DEBUG):
			print w, h
			print area
			cv2.imshow('image', single_char)
			cv2.waitKey(0)

		single_char = cv2.resize(single_char, CHAR_SIZE)
		single_char = single_char.reshape((1, 1000))
		single_char = np.float32(single_char)
		retval, char, neigh_resp, dists = model.findNearest(single_char, 1)
		results.append((x, y, int(char[0][0]), dists))

		
predict = ''
results.sort(cmp = custom_cmp)

for result in results:
	if (result[2] > 57):
		predict += chr(result[2] - 32)
	else:
		predict += chr(result[2])

print predict
