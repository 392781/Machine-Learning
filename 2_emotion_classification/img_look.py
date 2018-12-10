import numpy as np
import scipy.misc as smp
from PIL import Image
import sys
import os

x_train = np.load('data/x_train.npy')
#y_train = np.load('data/y_train.npy')

for i in range(0, x_train.shape[0]):
    dpath = os.path.abspath(os.curdir)
    dpath = os.path.join(dpath, "data_check\\")
    #y_val = int(y_train[i])
    img = smp.toimage(x_train[i].reshape(48,48))
    img = img.resize((256, 256), Image.ANTIALIAS)
    
    '''
    if (y_val == 0):
        path = "0_angry\\" + str(i) + '.png'
        dpath = os.path.join(dpath, path)
        img.save(dpath)
    elif (y_val == 1):
        path = "1_happy\\" + str(i) + '.png'
        dpath = os.path.join(dpath, path)
        img.save(dpath)
    else:
        path = "2_neutral\\" + str(i) + '.png'
        dpath = os.path.join(dpath, path)
        img.save(dpath)
    '''
    path = str(i) + '.png'
    dpath = os.path.join(dpath, path)
    img.save(dpath)
    
    s = str(i) + "/" + str(x_train.shape[0]) + " saved"
    sys.stdout.write(s)
    sys.stdout.flush()
    sys.stdout.write("\b" * len(s))
print("All images saved!")