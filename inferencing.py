import numpy as np
import tensorflow as tf
#from tensorflow import keras
from keras import Model

import matplotlib.pyplot as plt
import cv2
from random import randint
import subprocess

from keras.layers import Input,Conv2D,Subtract, Lambda
%matplotlib inline
from keras.models import load_model
import os
import keras.backend as K
import random
from random import randint



# save mask
cv2.imwrite(src+'mask'+str(pick)+'.png', mask)
mask_path = src+'mask'+str(pick)+'.png'


true_img_path =  src+'masked_img' +str(pick) + '.jpg'
true_img = cv2.imread(src+str(pick)+'.jpg')
plt.imshow(cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB))

#save asked image
masked_img = true_img.copy()
masked_img[mask > 0] = 255
cv2.imwrite(src+'masked_img'+str(pick)+'.jpg', masked_img)
masked_img_path = src+'masked_img'+str(pick)+'.jpg'
output_img_path= src+'output'+str(pick)+'.png'


# here should change ....too many times load model
subprocess.call(f"python test.py --image {masked_img_path} --mask {mask_path} \
--output {output_img_path} --checkpoint_dir model_logs/release_imagenet_256 ", shell=True)

output_img =  cv2.imread(output_img_path)
