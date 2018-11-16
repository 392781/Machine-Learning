import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

df_x_train = pd.read_csv('data/train_data.csv', header = None)
df_x_test = pd.read_csv('data/test_data.csv', header = None)
df_y_train = pd.read_csv('data/train_target.csv', header = None)

df_x_train = df_x_train.drop([12956, 15263, 2691, 600, 7879, 8178, 8274, 9484, 9755, 7094, 9583, 5373, 16167, 7602, 10468, 16154, 5085, 11253, 11640, 2639, 15352, 11255, 10888, 8470, 13273, 5006, 770, 1510, 11029, 15671, 7977, 4287, 8223, 2932, 12813, 1126, 5742, 2028, 1583, 4082])

df_y_train = df_y_train.drop([12956, 15263, 2691, 600, 7879, 8178, 8274, 9484, 9755, 7094, 9583, 5373, 16167, 7602, 10468, 16154, 5085, 11253, 11640, 2639, 15352, 11255, 10888, 8470, 13273, 5006, 770, 1510, 11029, 15671, 7977, 4287, 8223, 2932, 12813, 1126, 5742, 2028, 1583, 4082])

# CONSIDER USING STRATIFIED K FOLD
x_train, x_valid, y_train, y_valid = train_test_split(df_x_train,df_y_train, test_size = 0.1, random_state = 100)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

x_train = x_train.values
x_valid = x_valid.values
x_test = df_x_test.values
y_train = y_train.values
y_valid = y_valid.values

x_train = x_train.reshape(-1, 48, 48, 1)
x_valid = x_valid.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)


np.save(open('data/x_train.npy', 'wb'), np.array(x_train))
print("x_train saved")
np.save(open('data/y_train.npy', 'wb'), np.array(y_train))
print("y_train saved")
np.save(open('data/x_valid.npy', 'wb'), np.array(x_valid))
print("x_valid saved")
np.save(open('data/y_valid.npy', 'wb'), np.array(y_valid))
print("y_valid saved")
np.save(open('data/x_test.npy', 'wb'), np.array(x_test))
print("x_test saved")
np.save(open('data/class_weights.npy', 'wb'), np.array(class_weights))
print("class weights saved")
