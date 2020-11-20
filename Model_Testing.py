from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import cv2
import glob
import os


def test_sample(x):
    height, width = 120, 160
    output = "sample.mp4"
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    for frame in x_test[x]:
        out.write(frame)
    out.release()

    prediction = model.predict(x_train[x])

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('sample.mp4')

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video  file")

    # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # Display the resulting frame
            cv2.imshow(f"The predicted speed is {prediction} mph", frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            cap = cv2.VideoCapture('sample.mp4')

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


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
folder = "Some_images"
for i in range(len(os.listdir(folder))):
    filename = "frame" + str(i) + ".jpg"
    print(filename)
    data.append(asarray(Image.open(os.path.join(os.getcwd(),folder,filename))))

combined_data = []
for i in range(
        len(data) - window_size):  # Looping through all but the last window_size images to combine into one array
    combined_data.append(data[i:i + window_size])

# This removes the first bunch of labels because we use the last image of the window to determine speed
labels = asarray(labels[window_size:len(combined_data) + window_size])
print(len(combined_data), len(labels))
# Split the data
rm = 42
x_train, x_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.2, shuffle=False,
                                                    random_state=rm)

modelNum = "4"
model = keras.models.load_model("model" + modelNum + ".keras")
print(model.summary())

test_sample(0)
