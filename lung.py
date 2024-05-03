#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:46:11 2024

@author: DrVaniV
"""

from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from glob import glob
import matplotlib.pyplot as plt

# resizing the images to this size as per the the CNN variant requirements
IMAGE_SIZE =[224,224]
vgg = VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

for layer in vgg.layers:
    layer.trainable=False

folders = glob('Datasets/train/*')
x=Flatten()(vgg.output)
prediction = Dense(len(folders),activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
    loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=-2,zoom_range=2, 
horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('Datasets/train',target_size=(224,224),
                                                batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('Datasets/test',target_size=(224,224),
                                                 batch_size=32,class_mode='categorical')

r= model.fit(training_set,validation_data=test_set,epochs=5,steps_per_epoch=len(training_set),
                       validation_steps=len(test_set))
plt.plot(r.history['loss'],label='training loss')
plt.plot(r.history['val_loss'],label='validation loss')
plt.legend()
plt.show()
plt.savefig('LossVal_Loss')
plt.plot(r.history['accuracy'],label='training accuracy')
plt.plot(r.history['val_accuracy'],label='validation accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('model_vgg16.h5')


    