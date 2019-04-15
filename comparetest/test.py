import cv2
import os

test1="test4.png"
test2="test3.png"

source1=cv2.imread(test1)
source2=cv2.imread(test2)

#BGR --> HSV
conv1=cv2.cvtColor(source1, cv2.COLOR_BGR2HSV)
conv2=cv2.cvtColor(source2, cv2.COLOR_BGR2HSV)

hist1=cv2.calcHist([conv1], [0], None, [256], [0, 256])
hist2=cv2.calcHist([conv2], [0], None, [256], [0, 256])

h_dist = cv2.compareHist(hist1, hist2, 0)

print(h_dist)

