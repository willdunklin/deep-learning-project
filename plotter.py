#from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

modelToLoad = 4

log_data = pd.read_csv("model" + str(modelToLoad) + ".log", sep=',', engine='python')

plt.plot(log_data['loss'])
plt.plot(log_data['val_loss'])
plt.show()