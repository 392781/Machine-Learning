import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Reads CSV's into dataframes without headers
df_x_train = pd.read_csv('data/train_data.csv', header = None)
df_x_test = pd.read_csv('data/test_data.csv', header = None)
df_y_train = pd.read_csv('data/train_target.csv', header = None)

# Use scikit to split training data into training and validation sets
#x_train, x_valid, y_train, y_valid = train_test_split(df_x_train,df_y_train, test_size = 0.1, random_state = 100)

# Grab the dataframe values and make them into arrays
x_train = df_x_train.values
#x_valid = x_valid.values
x_test = df_x_test.values
y_train = df_y_train.values
#y_valid = y_valid.values

# Reshape training, validation, and test data to be the shape of the image
x_train = x_train.reshape(-1, 48, 48, 1)
#x_valid = x_valid.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)

# Save all arrays to directory to be loaded when running neural network
np.save(open('data/x_train.npy', 'wb'), np.array(x_train))
print("x_train saved")
np.save(open('data/y_train.npy', 'wb'), np.array(y_train))
print("y_train saved")
#np.save(open('data/x_valid.npy', 'wb'), np.array(x_valid))
#print("x_valid saved")
#np.save(open('data/y_valid.npy', 'wb'), np.array(y_valid))
#print("y_valid saved")
np.save(open('data/x_test.npy', 'wb'), np.array(x_test))
print("x_test saved")
