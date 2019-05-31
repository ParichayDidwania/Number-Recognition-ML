import tensorflow as tf
import cv2
import os
from PIL import Image
import imageio

dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/processed_images/'
dir2 = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/processed_images_2/'

for i in os.listdir(dir):
  img = cv2.imread(dir+i,0)
  backtorgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
  cv2.imwrite(dir2+i,backtorgb)
  print(imageio.imread(dir2+i).shape)


  
    
