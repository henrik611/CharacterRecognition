import sys
import cv2
import numpy as np

FILE_NAME = 'train_data/bienso' + sys.argv[1] + '.jpg'
PLATE_SIZE = (190 * 3, 140 * 3)
THRESHOLD_VALUE = 100
MIN_CHAR_WIDTH = 20
MAX_CHAR_WIDTH = 100
MIN_CHAR_HEIGHT = 140
MAX_CHAR_HEIGHT = 200
CHAR_SIZE = (20, 50)


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

cv2.imshow('image', thresh)
cv2.waitKey(0)

for contour in contours:
	[x, y, w, h] = cv2.boundingRect(contour)
	area = cv2.contourArea(contour, True)
	if area < 0 and MIN_CHAR_WIDTH < w and w < MAX_CHAR_WIDTH and MIN_CHAR_HEIGHT < h and h < MAX_CHAR_HEIGHT:
		single_char = thresh[y: y+ h, x: x + w]
		cv2.imshow('image', single_char)
		single_char = cv2.resize(single_char, CHAR_SIZE)
		single_char = single_char.reshape((1, 1000))
		key = cv2.waitKey(0)
		if (key == 32):
			continue
		else:
			f_samples = open('train_data/samples.txt', 'a')
			f_samples.write(' '.join(str(x) for x in single_char[0]))
			f_samples.write('\n')

			f_labels = open('train_data/labels.txt', 'a')
			f_labels.write(str(key))
			f_labels.write('\n')
