from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras import regularizers

import numpy as np

np.random.seed(0)

# Importing nparrays of training + validation + test data
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_valid = np.load('data/x_valid.npy')
y_valid = np.load('data/y_valid.npy')
x_test = np.load('data/x_test.npy')

# Changing target data from single value to categorical --> [x, y, z]
y_train = np_utils.to_categorical(y_train, 3)
y_valid = np_utils.to_categorical(y_valid, 3)

# Training image generator with presets to shift, flip, and shear
train_generator = ImageDataGenerator(width_shift_range = 0.1,
				height_shift_range = 0.1,
				horizontal_flip = True,
				shear_range = 0.2)

# Validation image generator with presets to shift, flip, and shear
valid_generator = ImageDataGenerator(width_shift_range = 0.1,
				height_shift_range = 0.1,
				horizontal_flip = True,
				shear_range = 0.2)

# I think this is needed for more advanced image generation
# Didn't end up using it in the end
train_generator.fit(x_train)
valid_generator.fit(x_valid)

# Beginning of model - Based on VGG19
# The model here was the one that gave me the score before the late submission
# Used 60 epochs:
#       1. 2 stack Conv2D 64
#       2. 2 stack Conv2D 128
#       3. 4 stack Conv2D 256
#       4. 4 stack Conv2D 1024
#       5. 2 fully connected layers 2048
#       6. 1 fully connected output layer
#
# The model below got me the late submission score of 0.83501
# It excluded the Conv2D 256 layers.  It ran for 100 epochs.
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
'''
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
'''
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# SGD optimizer - Found better results than adam
sgd = SGD(lr = 0.001, momentum = 0.9)

# Adam optimizer - Had this set up for testing, didn't use it much
adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)

# Compile model
model.compile(optimizer = sgd,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Begin training -> Using batch size 32 for training generator and validation generator
# Used 100 epochs since I felt confident that this was ok to protect against overtraining
model.fit_generator(train_generator.flow(x_train, y_train,
				batch_size = 32),
		steps_per_epoch = x_train.shape[0] // 32,
		epochs = 100,
		validation_data = valid_generator.flow(x_valid, y_valid, batch_size = 32),
		validation_steps = x_valid.shape[0] // 32) 

# Make predictions on test data
y_pred = model.predict(x_test)
print("Prediction done")

# Calculate categorical data back to regular single value data
result = np.argmax(y_pred, axis = 1)
print("Reults done")

# Saves the single value data to directory for processing
np.save(open('data/y_pred.npy', 'wb'), np.array(result))
print("Array saved")
