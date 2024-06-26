#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:37:58 2024

@author: DrVaniV
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('model_vgg16.h5')
img = image.load_img('val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

if(classes[0,0]== 1):
    print ("X-ray is Normal")
else:
    print ("X-ray is Abnormal and patient is affected by pneumonia")