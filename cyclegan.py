import numpy as np
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import os
import glob
import cv2



path = 'vangogh2vid/pytorch-CycleGAN-and-pix2pix/results/vangogh2photo_pretrained/test_latest/images/*_fake.png'
img_array = []
for image_file in glob.glob(path):
        image = cv2.imread(image_file)
        img_array.append(image)

h,w,l = image.shape
size = (w,h)

out = cv2.VideoWriter('project.avi', 0, 24, size)

for i in range(len(img_array)):
        out.write(img_array[i])
out.release()

print(len(img_array))


##vid = cv2.VideoCapture('avengers.mp4')

##count = 0
##while(vid.isOpened()):
##  ret, frame = vid.read()
##  frame = cv2.resize(frame, (256, 256))
##  #if cv2.waitKey(1) & 0xFF == ord('q'):
##  #  break
##  name = 'images/%.5d.jpg' % count
##  cv2.imwrite(name, frame)
##  count += 1
##
##vid.release()
##cv2.destroyAllWindows()
