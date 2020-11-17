from tensorflow import keras
from keras.layers import Dense, Conv3D, MaxPool3D, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import os

# set to true and specify model version number if you want to load a model
loadModel = False
modelNum = 4

# read model version from txt file
if not loadModel:
    with open('modelVersion.txt') as f:
        modelNum = f.read()
        f.close()

# iterate model number
if not loadModel:
    modelNum = str(int(modelNum) + 1)

print("Model version: ", modelNum)

# Data prep

# This is how many images to check each time
window_size = 8

labels = []
with open("train.txt") as f:
    for line in f:
        labels.append(line)
    for i in range(len(labels)):
        labels[i] = float(labels[i][:-2])



data = []
folder = "frames"
for picture in os.listdir(folder): # Some_images only contains 64 images, for testing
    data.append(asarray(Image.open(folder + "/" + picture)))

combined_data = []
for i in range(len(data) - window_size): # Looping through all but the last window_size images to combine into one array
    combined_data.append(data[i:i + window_size])

# This removes the first bunch of labels because we use the last image of the window to determine speed
labels = asarray(labels[window_size:])
print(len(combined_data), len(labels))
# Split the data
x_train, x_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.2, shuffle=False)

try:
    if loadModel and "model" + modelNum + ".keras" in os.listdir(os.getcwd()):
        #Load from a saved state
        model = keras.models.load_model("model" + modelNum + ".keras")
    else:
        # Defining the model
        start_filter = 64
        INPUT_SHAPE = (window_size, 120, 160, 3)
        model2 = Sequential()
        model2.add(Conv3D(start_filter, (3, 3, 3), padding="same", input_shape=INPUT_SHAPE))
        model2.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model2.add(Conv3D(2*start_filter, (3, 3, 3), padding="same"))
        model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(1, 2, 2)))
        model2.add(Conv3D(4*start_filter, (3, 3, 3), padding="same"))
        model2.add(Conv3D(4*start_filter, (3, 3, 3), padding="same"))
        model2.add(Conv3D(4*start_filter, (3, 3, 3), padding="same"))
        model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(1, 2, 2)))
        model2.add(Conv3D(8*start_filter, (3, 3, 3), padding="same"))
        model2.add(Conv3D(8*start_filter, (3, 3, 3), padding="same"))
        model2.add(Conv3D(8*start_filter, (3, 3, 3), padding="same"))
        model2.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        model2.add(Flatten())
        model2.add(Dense(1024, activation="relu"))
        model2.add(Dense(1024, activation="relu"))
        model2.add(Dense(1, activation="relu"))
        model2.compile(optimizer="adam", loss="MAE", metrics=["mae", "mse", "mape"])

    # Training the model
    model.fit(asarray(x_train), asarray(y_train), epochs=5, batch_size=1, verbose=1, validation_split=.2, shuffle=True, callbacks=[csv_logger])

    model.save("model" + modelNum + ".keras")

    # write new model number to file
    if not loadModel:
        with open('modelVersion.txt', 'w') as f:
            f.write(modelNum)
            f.close()

except KeyboardInterrupt:
    model.save("model" + modelNum + ".keras")
    if not loadModel:
        with open('modelVersion.txt', 'w') as f:
            f.write(modelNum)
            f.close()
