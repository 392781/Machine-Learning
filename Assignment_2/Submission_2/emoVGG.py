from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint

import numpy as np

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_valid = np.load('data/x_valid.npy')
y_valid = np.load('data/y_valid.npy')
x_test = np.load('data/x_test.npy')
class_weights = np.load('data/class_weights.npy')

y_train = np_utils.to_categorical(y_train, 3)
y_valid = np_utils.to_categorical(y_valid, 3)

train_generator = ImageDataGenerator(width_shift_range = 0.2,
				height_shift_range = 0.2,
				horizontal_flip = True,
				shear_range = 0.2,
				rescale = 1./255)

valid_generator = ImageDataGenerator(width_shift_range = 0.1,
				height_shift_range = 0.1,
				horizontal_flip = True)

train_generator.fit(x_train)
valid_generator.fit(x_valid)


def custom_model_1():
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

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

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
	#model.add(Dense(2048, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	#model.add(Dense(2048, activation='relu', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Dense(4096, activation='relu')) 
	model.add(Dropout(0.5))
	model.add(Dense(3, activation='softmax'))

	model.summary()
	return model
	

	
sgd = SGD(lr = 0.001, momentum = 0.9)
adam = Adam(lr = 0.0001)
model = custom_model_1()
#model.load_weights("checkpoints/weights_best.hdf5")

model.compile(optimizer = sgd,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

path = 'checkpoints/weights_best.hdf5'
checkpoint = ModelCheckpoint(path, save_best_only = True, mode = 'max', verbose = 1, monitor = 'val_acc')
callbacks_list = [checkpoint]

model.fit_generator(train_generator.flow(x_train, y_train,
				batch_size = 32),
		steps_per_epoch = x_train.shape[0] // 32,
		epochs = 40,
		validation_data = valid_generator.flow(x_valid, y_valid, batch_size = 32),
		validation_steps = x_valid.shape[0] // 32,
		class_weight = class_weights,
		callbacks = callbacks_list) 

y_pred = model.predict(x_test)
print("Prediction done")

result = np.argmax(y_pred, axis = 1)
print("Reults done")

np.save(open('data/y_pred.npy', 'wb'), np.array(result))
print("Array saved")
