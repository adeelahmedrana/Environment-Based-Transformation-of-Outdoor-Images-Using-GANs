# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:31:32 2022

@author: adeel
"""


from numpy import vstack
#from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
import numpy as np


# Use the saved cyclegan models for image translation
from instancenormalization import InstanceNormalization  
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()


# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/weights/g_model_AtoB_012000.h5', cust)
model_BtoA = load_model('C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/weights/g_model_BtoA_012000.h5', cust)


#Load a single custom image
test_image = load_img("C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/test/190.jpg")
test_image = img_to_array(test_image)
test_image_input = np.array([test_image])  # Convert single image to a batch.
test_image_input = (test_image_input - 127.5) / 127.5

# plot B->A->B (Photo to Monet to Photo)
monet_generated  = model_BtoA.predict(test_image_input)
photo_reconstructed = model_AtoB.predict(monet_generated)
show_plot(test_image_input, monet_generated, photo_reconstructed)

