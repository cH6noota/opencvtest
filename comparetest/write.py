import cv2
import os

test1="test1.png"

source1=cv2.imread(test1,0)


cv2.imwrite("gray_scale.png", source1)


