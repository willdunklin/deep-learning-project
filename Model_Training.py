import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv3D, MaxPool3D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
import os

#Data prep

labels = []
with open("train.txt") as f:
    for line in f:
        labels.append(line)
    for i in range(len(labels)):
        labels[i] = float(labels[i][:-2])

dataGen = ImageDataGenerator()
data = dataGen.flow_from_directory("Images_2")



INPUT_SHAPE = (16,160,120,3)
model2 = Sequential()
model2.add(Conv3D(32, (3,3,3), padding="same", input_shape=INPUT_SHAPE))
model2.add(MaxPool3D(pool_size=(1,2,2), strides=(1,2,2)))
model2.add(Conv3D(64, (3,3,3),padding="same"))
model2.add(MaxPool3D(pool_size=(2,2,2), strides=(1,2,2)))
model2.add(Conv3D(128, (3,3,3),padding="same"))
model2.add(Conv3D(128, (3,3,3),padding="same"))
model2.add(Conv3D(128, (3,3,3),padding="same"))
model2.add(MaxPool3D(pool_size=(2,2,2), strides=(2,2,2)))
model2.add(Conv3D(256, (3,3,3),padding="same"))
model2.add(Conv3D(256, (3,3,3),padding="same"))
model2.add(Conv3D(256, (3,3,3),padding="same"))
model2.add(MaxPool3D(pool_size=(2,2,2), strides=(2,2,2)))
model2.add(Conv3D(512, (3,3,3), padding="same"))
model2.add(MaxPool3D(pool_size=(2,2,2), strides=(2,2,2)))
model2.add(Flatten())
model2.add(Dense(1024, activation="relu"))
model2.add(Dense(1024, activation="relu"))
model2.add(Dense(1, activation="relu"))
model2.summary()