import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import cv2
import keras
from tensorflow.keras.models import Model, Sequential
import os
import seaborn as sns

from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from livelossplot.keras import PlotLossesCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger


print(os.listdir("../dhusor/archive/"))

SIZE = 224

train_images = []
train_labels = [] 
for directory_path in glob.glob("../dhusor/archive/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# test
test_images = []
test_labels = [] 
for directory_path in glob.glob("../dhusor/archive/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


x_train, x_test = x_train / 255.0, x_test / 255.0

from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

lr = ReduceLROnPlateau(monitor="val_loss", factor=0.85, patience=5, verbose=1)
cp = ModelCheckpoint(f'../dhusor/resnet50.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min", restore_best_weights=True)

model.fit(x_train, y_train_one_hot, batch_size=32, epochs=50, verbose=1, validation_split=0.2, callbacks=[es, cp, lr, PlotLossesCallback()])


score = model.evaluate(x_test, y_test_one_hot)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])


from sklearn import metrics
label_pred = model.predict(x_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

Y_test = np.argmax(y_test_one_hot, axis=1) # Convert one-hot to index

print(metrics.accuracy_score(Y_test, pred))

print(metrics.classification_report(Y_test, pred))

prediction_NN = model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)