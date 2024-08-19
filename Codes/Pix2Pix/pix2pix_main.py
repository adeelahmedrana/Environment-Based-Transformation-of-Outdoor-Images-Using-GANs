# https://youtu.be/6pUSZgPJ3Yg
"""
Satellite image to maps image translation ​using Pix2Pix GAN
 
Data from: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
Also find other datasets here: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
"""
TF_ENABLE_GPU_GARBAGE_COLLECTION= False
import glob
import os
from os import listdir
from numpy import asarray, load
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
physical_devices = tf.config.experimental.list_physical_devices('GPU')


if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


print('done')


# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		#map_img,sat_img = pixels[:, :256], pixels[:, 256:]
		sat_img,map_img = pixels[:, :256], pixels[:, 256:]
		#src_list.append(sat_img)
		#tar_list.append(map_img)
		src_list.append(map_img)
		tar_list.append(sat_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path_train = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/P2P datset/cleaned/resized/s2aut/'
path_test = 'C:/Users/adeel/Downloads/Pix2Pix/251_satellite_image_to_maps_translation/rain/test/'

#######################################################
path = path_train # To Train

#path = path_test  # To Test

########################################################

# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)


n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()

#######################################

from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#Define data
# load and prepare training images
data = [src_images, tar_images]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

####################################################################
from datetime import datetime 
start1 = datetime.now() 

train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1) 
#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

#####################################################################

#Test trained model on a few images...

from keras.models import load_model
from numpy.random import randint
model = load_model('model_070000.h5')

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()
	#pyplot.savefig('C:/Users/adeel/Python/Weights Plus Code Rain Removal/a.png')


def norm_gen_image(gen_image):
	# scale from [-1,1] to [0,1]
	image = (gen_image + 1) / 2.0
	return image

def plot2_images(gen_img, count):
	#images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (gen_image + 1) / 2.0
	#titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		#pyplot.subplot(1, 1, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		#pyplot.title('off')
	#pyplot.show()
	pyplot.savefig('C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/P2P datset/cleaned/resized/w2s/' + str(count) +'.png',bbox_inches='tight',pad_inches = 0)
	return()

#output_path = 'C:/Users/adeel/Python/Weights Plus Code Rain Removal/'



[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
print('The Value of Random selection is ')
print(ix)
count = 0
for repeat in range(len(X1)):
	
	src_image, tar_image = X1[[count]], X2[[count]]
# generate image from source
	gen_image = model.predict(src_image)
	gen_image_norm = (gen_image + 1) / 2.0

#print('here is the value of norm_gen_image')
#print(gen_image_norm)	
	plot2_images(gen_image,count)
	count = count + 1


[X1, X2] = dataset
print(dataset)
# select random example
ix = randint(0, len(X1), 1)
print(ix)
print('AHoye')
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)