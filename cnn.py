# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:08:31 2023

@author: USER
"""

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import CSVLogger
!pip install livelossplot -q
from livelossplot.keras import PlotLossesCallback

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import numpy as np
import os
np.random.seed(42)



from google.colab import drive
drive.mount('/content/drive')



from keras.utils import plot_model
plot_model(
    model, 
    to_file='model_arch.png', 
    show_shapes=True,
    show_layer_names=True
)



# Data augmentation
training_data_generator = ImageDataGenerator(
    rotation_range=30,
    zoom_range=[0.6, 1.4],
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255,)
validation_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True, seed=42)
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True, seed=42)   #42


for X_batch, y_batch in validation_generator:
    # Show 9 images
    plt.figure(figsize=(14,14))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i])
    # show the plot
    plt.show()
    break



lr = ReduceLROnPlateau(monitor="val_loss", factor=0.85, patience=5, verbose=1)
chk_point = ModelCheckpoint(f'./model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor="val_loss", patience=35, verbose=1, mode="min", restore_best_weights=True)

model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=350,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    callbacks=[lr, chk_point, es, PlotLossesCallback(), CSVLogger(TRAINING_LOGS_FILE, append=False, separator=";")], 
    #use_multiprocessing=True,
    #workers=2,
    verbose=1)

model.save('final_model.h5')



test_data_dir = '../input/accident-dataset-8k-resized/test'

if not os.path.exists('./test_images/acci'):
    os.makedirs('./test_images/acci')

if not os.path.exists('./test_images/non'):
    os.makedirs('./test_images/non')
    
probabilities = model.predict_generator(validation_generator)
for index, probability in enumerate(probabilities):
    image_path = test_data_dir + "/" + validation_generator.filenames[index]

    image = mpimg.imread(image_path)
    plt.imshow(image)

    if probability[0] > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% Accident")
        plt.savefig(f'./test_images/acci/{index}_{int(probability[0]*100)}.jpg', dpi=400)
    else:
        prob = (1-probability[0])*100
        plt.title("%.2f" % prob + "% No Accident")
        plt.savefig(f'./test_images/non/{index}_{int(prob)}.jpg', dpi=400)
        
    if index%20 == 0:
        plt.show()