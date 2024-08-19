# https://youtu.be/VzIO5_R9XEM
# https://youtu.be/2MSGnkir9ew
"""
Cycle GAN: Monet2Photo

Dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

"""

# monet2photo
from os import listdir
from numpy import asarray
from numpy import vstack
#from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)


# # dataset path
# path = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/train/'
# #path = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/test/'


# # load dataset A - Monet paintings SUMMER
# dataA_all = load_images(path + 'summer/')
# dataA = dataA_all
# print('Loaded summer: ', dataA_all.shape)




# from sklearn.utils import resample
# # #To get a subset of all images, for faster training during demonstration
# # dataA = resample(dataA_all, 
# #                  replace=False,     
# #                  n_samples=50,    
# #                  random_state=42) 

# # load dataset B - Photos WINTER
# dataB_all = load_images(path + 'autumn/')
# dataB = dataB_all
# print('Loaded Conversion Dataset: ', dataB_all.shape)

# # #Get a subset of all images, for faster training during demonstration
# # #We could have just read the list of files and only load a subset, better memory management. 
# # dataB = resample(dataB_all, 
# #                  replace=False,     
# #                  n_samples=50,    
# #                  random_state=42) 

# # plot source images
# # n_samples = 3
# # for i in range(n_samples):
# # 	plt.subplot(2, n_samples, 1 + i)
# # 	plt.axis('off')
# # 	plt.imshow(dataA[i].astype('uint8'))
# # # plot target image
# # for i in range(n_samples):
# # 	plt.subplot(2, n_samples, 1 + n_samples + i)
# # 	plt.axis('off')
# # 	plt.imshow(dataB[i].astype('uint8'))
# # plt.show()



# # load image data
# data = [dataA, dataB]

# print('Loaded', data[0].shape, data[1].shape)

# #Preprocess data to change input range to values between -1 and 1
# # This is because the generator uses tanh activation in the output layer
# #And tanh ranges between -1 and 1
# def preprocess_data(data):
# 	# load compressed arrays
# 	# unpack arrays
# 	X1, X2 = data[0], data[1]
# 	# scale from [0,255] to [-1,1]
# 	X1 = (X1 - 127.5) / 127.5
# 	X2 = (X2 - 127.5) / 127.5
# 	return [X1, X2]

# dataset = preprocess_data(data)

# from cycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# # define input shape based on the loaded dataset
# image_shape = dataset[0].shape[1:]
# # generator: A -> B
# g_model_AtoB = define_generator(image_shape)
# # generator: B -> A
# g_model_BtoA = define_generator(image_shape)
# # discriminator: A -> [real/fake]
# d_model_A = define_discriminator(image_shape)
# # discriminator: B -> [real/fake]
# d_model_B = define_discriminator(image_shape)
# # composite: A -> B -> [real/fake, A]
# c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# # composite: B -> A -> [real/fake, B]
# c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

# from datetime import datetime 
# start1 = datetime.now() 
# # train models
# train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=55)

# stop1 = datetime.now()
# #Execution time of the model 
# execution_time = stop1-start1
# print("Execution time is: ", execution_time)

# ############################################

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

# # load dataset
# A_data = resample(dataA_all, 
#                  replace=False,     
#                  n_samples=50,    
#                  random_state=42) # reproducible results

# B_data = resample(dataB_all, 
#                  replace=False,     
#                  n_samples=50,    
#                  random_state=42) # reproducible results

# A_data = (A_data - 127.5) / 127.5
# B_data = (B_data - 127.5) / 127.5


# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model("C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/weights/S2Sp/g_model_AtoB_132000.h5", cust)
model_BtoA = load_model("C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/weights/S2Sp/g_model_BtoA_132000.h5", cust)

# # plot A->B->A (Monet to photo to Monet)
# A_real = select_sample(A_data, 1)
# B_generated  = model_AtoB.predict(A_real)
# A_reconstructed = model_BtoA.predict(B_generated)
# show_plot(A_real, B_generated, A_reconstructed)
# # plot B->A->B (Photo to Monet to Photo)
# B_real = select_sample(B_data, 1)
# A_generated  = model_BtoA.predict(B_real)
# B_reconstructed = model_AtoB.predict(A_generated)
# show_plot(B_real, A_generated, B_reconstructed)

# # plot A->B->A (Monet to photo to Monet)
# A_real = select_sample(A_data, 1)
# B_generated  = model_AtoB.predict(A_real)
# A_reconstructed = model_BtoA.predict(B_generated)
# show_plot(A_real, B_generated, A_reconstructed)
# # plot B->A->B (Photo to Monet to Photo)
# B_real = select_sample(B_data, 1)
# A_generated  = model_BtoA.predict(B_real)
# B_reconstructed = model_AtoB.predict(A_generated)
# show_plot(B_real, A_generated, B_reconstructed)

# # ##########################
# def plot2_images(gen_img, count):
# 	#images = vstack((src_img, gen_img, tar_img))
# 	# scale from [-1,1] to [0,1]
# 	images = (gen_image + 1) / 2.0
# 	#titles = ['Source', 'Generated', 'Expected']
# 	# plot images row by row
# 	for i in range(len(images)):
# 		# define subplot
# 		#pyplot.subplot(1, 1, 1 + i)
# 		# turn off axis
# 		pyplot.axis('off')
# 		# plot raw pixel data
# 		pyplot.imshow(images[i])
# 		# show title
# 		#pyplot.title('off')
# 	#pyplot.show()
# 	pyplot.savefig('C:/Users/adeel/Python/Weights Plus Code Rain Removal/generated_rain/' + str(count) +'.jpg',bbox_inches='tight',pad_inches = 0)
# 	return()


# [dataA, dataB] = dataset
# # select random example
# ix = randint(0, len(dataA), 1)
# print('The Value of Random selection is ')
# print(ix)
# count = 0
# for repeat in range(len(dataA)):
	
# 	src_image, tar_image = dataA[[count]], dataB[[count]]
# # generate image from source
# 	gen_image = model.predict(src_image)
# 	gen_image_norm = (gen_image + 1) / 2.0

# #print('here is the value of norm_gen_image')
# #print(gen_image_norm)	
# 	plot2_images(gen_image,count)
# 	count = count + 1

def plot2_images(a2b, count):
	#images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (a2b + 1) / 2.0
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
	pyplot.savefig('C:/Users/adeel/OneDrive/Desktop/TUI/Project/CycleGAN/count/' + str(count) +'.jpg',bbox_inches='tight',pad_inches = 0)
	return()


path_test = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/test/'
dataA_test = load_images(path_test)
dataA = dataA_test
print (dataA)

# dataA_test_image_input = np.array([dataA])  # Convert single image to a batch.
# dataA_test_image_input = (test_image_input - 127.5) / 127.5

path_test = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/test/'

#Load a single custom image
count = 0
for repeat in range(len(dataA)):
	test_image = load_img(path_test + str(count) +'.jpg')
	test_image = img_to_array(test_image)
	test_image_input = np.array([test_image])  # Convert single image to a batch.
	test_image_input = (test_image_input - 127.5) / 127.5
	a2b = model_AtoB.predict(test_image_input)
	b2a  = model_BtoA.predict(a2b)
	
	#show_plot(test_image_input, a2b, b2a)
	plot2_images(a2b,count)
	count=count+1





# plot B->A->B (Photo to Monet to Photo)


# test_image_input = np.array([test_image])  # Convert single image to a batch.
# test_image_input = (test_image_input - 127.5) / 127.5

# # plot B->A->B (Photo to Monet to Photo)
# monet_generated  = model_BtoA.predict(test_image_input)
# photo_reconstructed = model_AtoB.predict(monet_generated)
# show_plot(test_image_input, monet_generated, photo_reconstructed)

# count = 0
# for repeat in range(len(test_image)):
# 	test_image_input = np.array([test_image])  # Convert single image to a batch.
# 	test_image_input = (test_image_input - 127.5) / 127.5
# 	monet_generated  = model_BtoA.predict(test_image_input)
# 	photo_reconstructed = model_AtoB.predict(monet_generated)
# 	show_plot(test_image_input, monet_generated, photo_reconstructed)
	
# 	count = count + 1