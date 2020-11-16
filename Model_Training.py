import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv3D, MaxPool3D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy import asarray
import os

# Data prep

# This is how many images to check each time
window_size = 16

labels = []
with open("train.txt") as f:
    for line in f:
        labels.append(line)
    for i in range(len(labels)):
        labels[i] = float(labels[i][:-2])

# This removes the first bunch of labels because we use the last image of the window to determine speed
labels = labels[window_size:64] # Remove 64 when testing on full dataset

data = []
for picture in os.listdir("Some_images"): # Some_images only contains 64 images, for testing
    data.append(asarray(Image.open("Some_images/" + picture)))

combined_data = []
for i in range(len(data) - 16): # Looping through all but the last 16 images to combine into one array
    combined_data.append(data[i:i + window_size])


# Defining the model
INPUT_SHAPE = (window_size, 120, 160, 3)
model2 = Sequential()
model2.add(Conv3D(32, (3, 3, 3), padding="same", input_shape=INPUT_SHAPE))
model2.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model2.add(Conv3D(64, (3, 3, 3), padding="same"))
model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(1, 2, 2)))
model2.add(Conv3D(128, (3, 3, 3), padding="same"))
model2.add(Conv3D(128, (3, 3, 3), padding="same"))
model2.add(Conv3D(128, (3, 3, 3), padding="same"))
model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model2.add(Conv3D(256, (3, 3, 3), padding="same"))
model2.add(Conv3D(256, (3, 3, 3), padding="same"))
model2.add(Conv3D(256, (3, 3, 3), padding="same"))
model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model2.add(Conv3D(512, (3, 3, 3), padding="same"))
model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
model2.add(Flatten())
model2.add(Dense(1024, activation="relu"))
model2.add(Dense(1024, activation="relu"))
model2.add(Dense(1, activation="relu"))
model2.compile(optimizer="adam", loss="MSE", metrics=["accuracy"])

# Training the model
model2.fit([combined_data], labels, epochs=1, verbose=1)
