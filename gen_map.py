import numpy as np
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
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import h5py
import imageio

# Load the latest saved model (model_name.h5)
model = load_model('model_name.h5')

xtl = []

# path of the image
path = "codes_GRBM/codes_GRBM/target_images/3.jpg"
img = keras.utils.load_img(path)
img = keras.utils.img_to_array(img)

# CONSTANTS
imagewidth = 256
imageheight = 256
windowsize = 32
stride = 2

# iterations
iterations_w = (imagewidth - windowsize)/stride + 1
iterations_h = (imageheight - windowsize)/stride + 1


# SCORES ARRAY INSTEAD OF THIS
score = np.random.uniform(low=0, high=1, size=(iterations_h*iterations_w,))

# scores' data
lowest_score = np.amin(score)
highest_score = np.amax(score)
step_size = (highest_score - lowest_score)/5

# image for heatmap
newim = np.zeros((3, 256, 256), dtype='float32')

# intensity control factor (to change contrast)
icf = 20

# increment counter
l = 0


def rgb(value):
    # minimum, maximum = lowest_score, highest_score
    ratio = 2 * (value - lowest_score) / (highest_score - lowest_score)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


# define standard constant colors
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


# set tolerance
EPSILON = (highest_score - lowest_score)/5


# processing image data
def rgb(val):
    fi = float(val - lowest_score) / \
        float(highest_score - lowest_score) * (len(colors)-1)
    i = int(fi)
    f = fi - i
    if f < EPSILON:
        return colors[i]
    else:
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))


""" 
		Color coding scheme
		Pixel intensity is set according to the memorability score of a window stride

"""
for i in range(int(iterations_w)):
    for j in range(int(iterations_h)):
        # (r,g,b)
        a = rgb(score[l])

        # defining intervals for color coding
        if(score[l] >= lowest_score and score[l] < lowest_score+step_size):
            newim[0][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = a[0]
            newim[1][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = a[1]
            newim[2][(j*stride):(j*stride)+windowsize, (i*stride):(i *
                                                                   stride)+windowsize] = a[2]  # (score[l] - lowest_score)*icf

        elif(score[l] >= lowest_score+step_size and score[l] < lowest_score+2*step_size):
            newim[0][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 0
            newim[1][(j*stride):(j*stride)+windowsize, (i*stride)
                      :(i*stride)+windowsize] = (score[l] - lowest_score)*icf
            newim[2][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 1

        elif(score[l] >= lowest_score+2*step_size and score[l] < lowest_score+3*step_size):
            newim[0][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 0
            newim[1][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 1
            newim[2][(j*stride):(j*stride)+windowsize, (i*stride)
                      :(i*stride)+windowsize] = (highest_score - score[l])*icf

        elif(score[l] >= lowest_score+3*step_size and score[l] < lowest_score+4*step_size):
            newim[0][(j*stride):(j*stride)+windowsize, (i*stride)
                      :(i*stride)+windowsize] = (score[l] - lowest_score)*icf
            newim[1][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 1
            newim[2][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 0

        else:
            # if(score[l]>=lowest_score+4*step_size and score[l]<=lowest_score+5*step_size)
            newim[0][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 1
            newim[1][(j*stride):(j*stride)+windowsize, (i*stride)
                      :(i*stride)+windowsize] = (highest_score - score[l])*icf
            newim[2][(j*stride):(j*stride)+windowsize,
                     (i*stride):(i*stride)+windowsize] = 0
        l += 1


# changing dimension to display image
newim = np.rollaxis(newim, 0, 3)


"""
 		FOR SAVING HEATMAP NEW IMAGE
	    'outfile.jpg' is the name of map image

"""
# scipy.misc.imsave('outfile.jpg', newim) // deprecated
imageio.imwrite('outfile.jpg', newim)

# Load the image and the image map
img_map = keras.utils.load_img('outfile.jpg')
orig_image = keras.utils.load_img(path)

# display the image
plt.imshow(Image.blend(orig_image, img_map, alpha=.9))
plt.show()
