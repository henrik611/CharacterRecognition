import cv2
import numpy as np

samples = np.loadtxt('data/samples.txt',np.float32)
labels = np.loadtxt('data/labels.txt',np.float32)
labels = labels.reshape((labels.size,1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, labels)

ret, results, neighbours, dist = model.findNearest(np.array(samples[0], ndmin = 2), 1)
print results
