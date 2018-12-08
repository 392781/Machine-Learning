'''
RUNNING PROCESS:
    python gen_data.py 
   *python model.py
    python predict.py
    python masks2csv.py
'''

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from skimage.transform import resize
from skimage.io import imsave

from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

# Unet from the tutorial, mostly the same but activation is changed to relu
def starter_unet(height = 256, width = 256, channels = 3):
	inputs = Input((height, width, channels))
    # Resizing pixels to be represented between 0 and 1
	s = Lambda(lambda x: x / 255.)(inputs)

	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
	
	u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
	
	u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
	
	u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.1) (c8)
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
	
	u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
	
	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
	
	model = Model(inputs=[inputs], outputs=[outputs])
	model.summary()
	
	return model

# Unet I found online (Though it was also in the picture of the starter guide)
# Modified to not have kernel regularization as that slowed training down a lot
def custom_unet(input_height = 256, input_width = 256, channels = 3):
	inputs = Input((input_height, input_width, channels))
    # Resizing pixels to be represented between 0 and 1
	s = Lambda(lambda x: x / 255.)(inputs)	

	c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(s)
	c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(c1)
	p1 = MaxPooling2D((2, 2), name = 'block1_pool')(c1)
	
	c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(p1)
	c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(c2)
	p2 = MaxPooling2D((2, 2), name = 'block2_pool')(c2)

	c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(p2)
	c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(c3)
	p3 = MaxPooling2D((2, 2), name = 'block3_pool')(c3)

	c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(p3)
	c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(c4)
	c4 = Dropout(0.5)(c4)
	p4 = MaxPooling2D((2, 2), name = 'block4_pool')(c4)
	
	c5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1')(p4)
	c5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2')(c5)
	c5 = Dropout(0.5)(c5)
	
	u6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block6_up')(UpSampling2D(size = (2, 2))(c5))
	m6 = concatenate([c4, u6], axis = 3)
	c6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block6_conv1')(m6)
	c6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block6_conv2')(c6)

	u7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block7_up')(UpSampling2D(size = (2, 2))(c6))
	m7 = concatenate([c3, u7], axis = 3)
	c7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block7_conv1')(m7)
	c7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block7_conv2')(c7)
	
	u8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block8_up')(UpSampling2D(size = (2, 2))(c7))
	m8 = concatenate([c2, u8], axis = 3)
	c8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block8_conv1')(m8)
	c8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block8_conv2')(c8)

	u9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block9_up')(UpSampling2D(size = (2, 2))(c8))
	m9 = concatenate([c1, u9], axis = 3)
	c9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block9_conv1')(m9)
	c9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block9_conv2')(c9)
	o = Conv2D(1, (1, 1), activation = 'sigmoid')(c9)

	model = Model(inputs = [inputs], outputs = [o])

	return model


# Dice coefficient, found online
# Smooth set to 1e6 to allow for nice percentage values	
smooth = 1e6
def dice_coef(y_true, y_pred):
	tr = K.flatten(y_true)
	pr = K.flatten(y_pred)
	return (2. * K.sum(tr * pr) + smooth) / (K.sum(tr) + K.sum(pr) + smooth)

# Loading preprocessed numpy arrays 
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_test = np.load('data/x_test.npy')
x_valid = np.load('data/x_valid.npy')
y_valid = np.load('data/y_valid.npy')

# Generators I used during testing.  NOT used to generate my top submission.
'''
train_gen = ImageDataGenerator(width_shift_range = 0.05,
		height_shift_range = 0.05,
		horizontal_flip = True,
		shear_range = 0.05)

valid_gen = ImageDataGenerator(horizontal_flip = True)

train_gen.fit(x_train)
valid_gen.fit(x_valid)
'''

# Adam optimizer
adam = Adam(lr = 1e-4)

# Loading + compiling unet using dice_coef as metric
model = custom_unet()
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = [dice_coef])
model.summary()

# Callbacks used.  All depending on validation dice coefficient.
earlystopper = EarlyStopping(monitor = 'val_dice_coef', 
        patience = 5, 
        verbose = 1, 
        mode = 'max')
checkpointer = ModelCheckpoint('my_model.h5', 
        monitor = 'val_dice_coef', 
        verbose = 1, 
        save_best_only = True, 
        mode = 'max')

# ImageDataGenerator fitting used during testing.  NOT used to generate my top submission.
'''
model.fit_generator(train_gen.flow(x_train, y_train,
		batch_size = 16),
		steps_per_epoch = x_train.shape[0] // 16,
		epochs = 50,
		validation_data = valid_gen.flow(x_valid, y_valid, batch_size = 16),
		validation_steps = x_valid.shape[0] // 16,
		callbacks = [earlystopper, checkpointer])
'''

# Model fitting
model.fit(x_train, y_train, 
        validation_data = (x_valid, y_valid),
		batch_size = 16,
		epochs = 50,
		verbose = 1,
		callbacks = [earlystopper, checkpointer])

