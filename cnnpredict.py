# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:12:08 2022

@author: HP
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

image_path = 'D:/THESIS/Classification final/6.JPG'
model = load_model('D:/THESIS/Classification final/Models/15M Param/final_model.h5')

img = cv2.imread(image_path)
img = cv2.resize(img, (224,224))
img = np.array(img).astype('float32')/255.0
img = np.expand_dims(img, axis=0)
prob = model.predict(img)

image = mpimg.imread(image_path)
plt.imshow(image)

if prob > 0.5:
    plt.title("%.2f" % (prob*100) + "% No Accident")
else:
    prob = (1-prob)*100
    plt.title("%.2f" % prob + "% Accident")
        
plt.savefig(f'D:/THESIS/Classification final/Dataset/Prediction2/{int(prob*100)}.jpg', dpt=1400)
plt.show()