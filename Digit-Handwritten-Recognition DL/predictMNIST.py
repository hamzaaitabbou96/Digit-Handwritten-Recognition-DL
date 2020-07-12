# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:22:50 2020

@author: Achraf
"""


import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

loaded_model =load_model('modelfinal1.h5')

image_path="testn.png"

img = image.load_img(image_path, target_size=(28, 28, 1), color_mode='grayscale')

plt.imshow(img)

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)

result=loaded_model.predict_classes(img)
print(result)



#image_path="testn.png"
#img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
#plt.imshow(img)
#img = np.expand_dims(img, axis=0)
#result=loaded_model.predict_classes(img)
