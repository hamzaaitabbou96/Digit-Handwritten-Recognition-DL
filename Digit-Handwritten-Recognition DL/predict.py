# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:40:53 2020

@author: hamzaa_aitabbou
"""


import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

loaded_model =load_model('models/model98.93.h5')

image_path="test2.png"

img = image.load_img(image_path, target_size=(28, 28, 1), color_mode='grayscale')

plt.imshow(img)

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

result=loaded_model.predict_classes(img)
print('-_-')
print(result)


