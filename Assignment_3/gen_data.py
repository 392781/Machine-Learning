'''
RUNNING PROCESS:
   *python gen_data.py
    python model.py
    python predict.py
    python masks2csv.py
'''

import os
from tqdm import tqdm
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label

# Setting intended sizes to resize data with
HEIGHT = 256
WIDTH = 256
CHANNELS = 3

# Grabbing the path of respective directories
training_x_path = './data/training_images/'
training_y_path = './data/training_masks/'
test_path = './data/testing_images/'
valid_x_path = './data/validation_images/'
valid_y_path = './data/validation_masks/'

# Grabbing the labels (and by consequences size) of each directory
train_x = next(os.walk(training_x_path))[2]
train_y = next(os.walk(training_y_path))[2]
testing = next(os.walk(test_path))[2]
valid_x = next(os.walk(valid_x_path))[2]
valid_y = next(os.walk(valid_y_path))[2]

# Initializing arrays for training, validation, and testing data
x_train = np.zeros((len(train_x), HEIGHT, WIDTH, CHANNELS), dtype = np.uint8)
y_train = np.zeros((len(train_y), HEIGHT, WIDTH, 1), dtype = np.uint8)
x_test = np.zeros((len(testing), HEIGHT, WIDTH, CHANNELS), dtype = np.uint8)
x_valid = np.zeros((len(valid_x), HEIGHT, WIDTH, CHANNELS), dtype = np.uint8)
y_valid = np.zeros((len(valid_y), HEIGHT, WIDTH, 1), dtype = np.uint8)

# Resizing training images
print("\nResizing training images")

for n, id_ in tqdm(enumerate(train_x), total = len(train_x)):
	path = training_x_path + id_
	img = imread(path)
	img = resize(img, (HEIGHT, WIDTH), mode = 'constant', preserve_range = True)
	x_train[n] = img
    
# Resizing + processing training masks to have pixel values between 0 and 1
print("Resizing training masks")

for n, id_ in tqdm(enumerate(train_y), total = len(train_y)):
	path = training_y_path + id_
	img = imread(path)
	img = np.expand_dims(resize(img, (HEIGHT, WIDTH), mode = 'constant', preserve_range = True), axis = -1)
	img = np.round(img/255.)
	y_train[n] = img
    
# Resizing testing images
print("Resizing test images")

for n, id_ in tqdm(enumerate(testing), total = len(testing)):
	path = test_path + id_
	img = imread(path)
	img = resize(img, (HEIGHT, WIDTH), mode = 'constant', preserve_range = True)
	x_test[n] = img

# Resizing validation images
print("Resizing validation images")

for n, id_ in tqdm(enumerate(valid_x), total = len(valid_x)):
	path = valid_x_path + id_
	img = imread(path)
	img = resize(img, (HEIGHT, WIDTH), mode = 'constant', preserve_range = True)
	x_valid[n] = img

# Resizing + processing validation masks to have pixel values between 0 and 1
print("Resizing validation masks")

for n, id_ in tqdm(enumerate(valid_y), total = len(valid_y)):
	path = valid_y_path + id_
	img = imread(path)
	img = np.expand_dims(resize(img, (HEIGHT, WIDTH), mode = 'constant', preserve_range = True), axis = -1)
	img = np.round(img/255.)
	y_valid[n] = img
 
# Saving the processed numpy arrays to directory for later use 
np.save(open('data/x_train.npy', 'wb'), x_train)
print("x_train saved")
np.save(open('data/y_train.npy', 'wb'), y_train)
print("y_train saved")
np.save(open('data/x_test.npy', 'wb'), x_test)
print("x_test saved")
np.save(open('data/x_valid.npy', 'wb'), x_valid)
print("x_valid saved")
np.save(open('data/y_valid.npy', 'wb'), y_valid)
print("y_valid saved")
