import sys
import os

import cv2
import numpy as np

cur = os.getcwd()
sys.path.insert(0,os.path.join(cur,'../lib'))

from utils import read_intri,undistortion

yml_file = '../data/output/intri.yml'
img_path = '../output/IMG_20210917_105453.jpg'

cam_dict = read_intri(yml_file)

k = cam_dict["01"]['K']
dist = cam_dict["01"]['dist']

img = cv2.imread(img_path)
new_img = undistortion(img,k,dist)

d_img = np.hstack([img,new_img])

cv2.imshow("undistortion",d_img)
cv2.waitKey()