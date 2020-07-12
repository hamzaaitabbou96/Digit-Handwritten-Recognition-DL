# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:40:53 2020

@author: hamzaa_aitabbou
"""

import  numpy  as np	
import keras
from keras.datasets import mnist

from  keras.models  import  Sequential

from  keras.layers  import  Conv2D,  MaxPooling2D
from  keras.layers  import  Flatten,  Dense

(x_train,  y_train),  (x_test,  y_test)  =  mnist.load_data()
x_train  =  x_train.reshape(60000,28,28,1) 
x_test  =  x_test.reshape(10000,28,28,1)

y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10)

model  =  Sequential()

model.add(Conv2D(32,  (3,  3),  input_shape  =  (28,  28,  1),  activation  =  'relu'))
model.add(MaxPooling2D(pool_size  =  (2,  2)))

model.add(Conv2D(32,  (3,  3),  activation  =  'relu'))
model.add(MaxPooling2D(pool_size  =  (2,  2)))


model.add(Flatten())

model.add(Dense(units  =  128,  activation  =  'relu'))
model.add(Dense(units  =  10,  activation  =  'softmax'))

model.compile(optimizer  =  'adam',  loss  =  'categorical_crossentropy',  metrics  =  ['ac curacy'])
model.fit(x_train,  y_train,batch_size=128, epochs=25, verbose=1,validation_data=(x_test,  y_test))


score = model.evaluate(x_test, y_test, verbose=0) 
print('Test  loss:',  score[0])
print('Test  accuracy:',  score[1])

model.save('models/model98.93.h5') 
model.save_weights('models/weights_model98.93')
