"""
	In this file we define a custom CNN model without using transfer learning technique.

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
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def name_image(name="1"):
	if len(name)>8:
		return name
	else:
		for i in range(8-len(name)):
			name = "0" + name
		return name


with open("train_1.txt") as f:
	content = f.readlines()
print len(content)


# Image constants
IMAGE_WIDTH  = 256
IMAGE_HEIGHT = 256

# Data Constants defining train-test split 
train_start  = 1
train_end    = 43001
val_end      = 47550


# Batch image generator that returns the training data in batches
def gen(batch_size=1, flag='train'):
	if flag=='train':
		start = train_start/batch_size
		end   = train_end/batch_size
	else:
		start = train_end/batch_size
		end   = val_end/batch_size
	x_train = np.zeros((batch_size, IMAGE_WIDTH,IMAGE_HEIGHT,3), dtype='float32')
	y_train = np.zeros((batch_size, 1), dtype='float32')
	
	while True:
		for i in range(start, end):
			for j in range(batch_size):
				if(len(content[i])>23):
					y_train[j,:] = float(content[i*batch_size+j][content[i*batch_size+j].find(" ")+1:])
					path = "pre/"+content[i*batch_size+j][:content[i*batch_size+j].find(" ")]
				else:
					y_train[j,:] = float(content[i*batch_size+j][13:])
					path = "lamem/images/"+content[i*batch_size+j][:12]					

				# preprocess the image
				img = image.load_img(path)
				img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
				img = image.img_to_array(img)
				img = preprocess_input(img, mode='tf')
				x_train[j, :img.shape[0], :img.shape[1], :] = img
			
			# return train data
			yield x_train, y_train


# Designing a new CNN model from scratch
def baseline_model():
	my_model = Sequential()
	
	# Convolution layers		
	my_model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,3), activation='relu'))
	my_model.add(Conv2D(32, (3, 3), activation='relu'))
	my_model.add(MaxPooling2D(pool_size=(2, 2)))
	
	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(MaxPooling2D(pool_size=(2, 2)))

	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(MaxPooling2D(pool_size=(2, 2)))

	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(Conv2D(64, (3, 3), activation='relu'))
	my_model.add(MaxPooling2D(pool_size=(2, 2)))
	
	my_model.add(Conv2D(32, (3, 3), activation='relu'))
	my_model.add(Conv2D(32, (3, 3), activation='relu'))
	my_model.add(MaxPooling2D(pool_size=(2, 2)))

	# Dropout and Flatten layer	
	my_model.add(Dropout(0.2))
	my_model.add(Flatten())

	# Fully Connected layers
	my_model.add(Dense(128, activation='relu'))
	my_model.add(Dropout(0.2))
	my_model.add(Dense(64, activation='relu'))
	my_model.add(Dense(1, activation='sigmoid'))
	my_model.summary()

	# Compile model
	my_model.compile(loss='mean_squared_error', optimizer='sgd')

	return my_model


# Initiate base model
model = baseline_model()


# Training parameters
batch_size = 16
EPOCHS = 100


# Callback function to save the epoch with least val_loss
filepath       = "weights.best.hdf5"
checkpoint     = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')


# finally, fit the model to data
model.fit_generator( gen(batch_size=batch_size),
					 steps_per_epoch = (train_end - train_start)/batch_size,
					 epochs = EPOCHS,
					 callbacks = [checkpoint],
					 validation_data = gen(batch_size=batch_size, flag='val'),
					 validation_steps = (val_end - train_end)/batch_size
					 )

model.save('lamem_trained_callback_augmented.h5')
