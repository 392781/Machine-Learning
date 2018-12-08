'''
RUNNING PROCESS:
    python gen_data.py
    python model.py
   *python predict.py
    python masks2csv.py
'''

import os
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from keras.models import load_model
from keras import backend as K

# Using dice coefficient to load model
smooth = 1e6
def dice_coef(y_true, y_pred):
        tr = K.flatten(y_true)
        pr = K.flatten(y_pred)
        return (2. * K.sum(tr * pr) + smooth) / (K.sum(tr) + K.sum(pr) + smooth)

# Loading testing array 
x_test = np.load('./data/x_test.npy')
# Loading model with saved best weights
model = load_model('my_model.h5', custom_objects = {'dice_coef': dice_coef})
# Predicting on testing images
y_pred = model.predict(x_test, verbose = 1)
# Threshold cutoff for pixels (>0.35 = 1, <0.35 = 0)
y_pred_t = (y_pred > 0.35).astype(np.uint8)
print("Predictions done")

# Reading the names of the testing images in order to properly name each image
test_path = 'data/testing_images/'
test_ids = next(os.walk(test_path))[2]
# Resizing images back to original size and saving them to directory
for i in range(len(y_pred_t)):
        img = np.squeeze(y_pred_t[i])
        img = resize(img, (250, 250), mode = 'constant', preserve_range = True)
        imsave('./results/' + 'test_mask_' + test_ids[i][9:], img)
