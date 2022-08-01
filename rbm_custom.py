"""
		Regression using RBM 
"""

import csv
import numpy as np
import pickle
import scipy
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as plt
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import itertools as it
from matplotlib.pyplot import figure


# training set nomenclature
def name_image(name=1):
    if len(name) > 8:
        return name
    else:
        for i in range(8-len(name)):
            name = "0"+name
        return name


def gen(batch_size=1, flag='train'):
    if flag == 'train':
        start = 1
        end = 3
    else:
        start = 3
        end = 4
    x_train = np.zeros((batch_size, 256*256), dtype='float32')
    y_train = np.zeros((batch_size, 1), dtype='float32')
    while True:
        for i in range(start, end):
            for j in range(batch_size):
                with open("scores.csv", "rb") as f:
                    reader = csv.reader(f)
                    y_train[j, :] = float(list(reader)[j][0])
                path = "codes_GRBM/codes_GRBM/target_images/"+str(j)+".jpg"
                img = keras.utils.load_img(path, grayscale=True)
                img = keras.utils.img_to_array(img)
                img = preprocess_input(img, mode='tf')
                x_train[j, :] = img.reshape(256*256)
            yield x_train, y_train


def baseline_model():
    my_model = Sequential()
    my_model.add(Dense(4, input_dim=256*256, activation='relu'))
    my_model.add(Dense(1, activation='sigmoid'))
    my_model.summary()
    # Compile model
    my_model.compile(loss='mean_squared_error', optimizer='sgd')
    return my_model


model = baseline_model()

batch_size = 4

EPOCHS = 1

history = model.fit(gen(batch_size=batch_size),
                    steps_per_epoch=2000/batch_size,
                    # steps_per_epoch = 10,
                    epochs=EPOCHS,
                    callbacks=None,
                    validation_data=gen(batch_size=batch_size, flag='val'),
                    validation_steps=222/batch_size
                    # validation_steps = 2
                    )

model.save('mdh.h5')

figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plot.png')
plt.show()
