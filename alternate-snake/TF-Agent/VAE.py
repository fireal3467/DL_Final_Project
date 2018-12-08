import tensorflow as tf
layers = tf.layers
import numpy as np
from scipy.misc import imsave
import os
import argparse

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


N_Z = 3200
BATCH_SIZE = 50


class Model:
    def __init__(self, inputs, outputs):
        self.images = inputs
        self.labels = outputs

        self.z = encoder(inputs)
        self.o_decoded = o_decoder(z)
        self.kernels = decoder(z)


    def preprocess(im1, im2):
    	'''
    	-- The input to the network in training is
		-- {I1 - mean, I2 - mean}
		-- And the output to the network in trainig is
		-- (I2 - I1) * 100
		--
		-- Therefore we need the following preproces and
		-- postprocess step
    	'''

    	pass

    def encoder(self, x):
        '''
        Input: [batch_size , 128, 128, 6] 3 layers for original image RBG, 3 for 'motion image' RGB 

        "The motion encoder takes the current frame as input, in addition to the motion image, 
        	so that it can learn to model the conditional variational distribution (qθ(·) in Eq. (5))."
        output: [Batch Size, N_Z]
        '''
        
        # Conv 0: [_, 128, 128, 6]
        output = tf.layers.conv2d(x, filters=96,  kernel_size=(5,5), padding='SAME')
        output = tf.layers.batch_normalization(output)

		# Conv 1: [_, 128, 128, 96]
        output = tf.layers.conv2d(output, filters=96,  kernel_size=(5,5), padding='SAME')
        output = tf.layers.batch_normalization(output)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

		# Conv 2: [_, 64, 64, 96]
        output = tf.layers.conv2d(output, filters=126,  kernel_size=(5,5), padding='SAME')
        output = tf.layers.batch_normalization(output)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

		# Conv 3: [_, 16, 16, 126]
        output = tf.layers.conv2d(output, filters=126,  kernel_size=(5,5), padding='SAME')
        output = tf.layers.batch_normalization(output)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

        # Conv 4: [_, 8, 8, 126]
        output = tf.layers.conv2d(output, filters=256,  kernel_size=(5,5), padding='SAME')
        output = tf.layers.batch_normalization(output)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)

		# Conv 5: [_, 4, 4, 256]
        output = tf.layers.conv2d(output, filters=256,  kernel_size=(5,5), padding='SAME')

        # ...

        # Conv 6: [_, 5, 5, 256]

        #what is the shape at the end of this? 256,5,5. The thing paper says it should be 256,5,5?
        reshaped = tf.reshape(output, [-1, 6400])

        #Split the thing into two [batchsize, 3200] vectors to get a mean and variance vector
        mean_vector, variance_vector = tf.split(reshaped, num_or_size_splits=2, 1)

        mu = tf.layers.dense(mean_vector, N_Z)
        #Q: Why is this log_sigma?
        log_sigma = tf.layers.dense(mean_vector, N_Z)


        #take a sample from random normal here
        eps = tf.random_normal(shape=[-1, N_Z], mean =0., std=1.)

        return mu + tf.exp(log_sigma/2) * eps

    def decoder(z):
        '''
        input: z of size [BATCH_SIZE, 3200]
        '''

        output = tf.reshape(z, [BATCH_SIZE,5,5,128])
        #Q: Honestly, not sure if this shape will work out rn
        output = tf.layers.conv2d(output, filters=128, kernel_size=(5,5), padding='SAME')
        output = tf.layers.conv2d(output, filters=128, kernel_size=(5,5), padding='SAME')
        split1, split2, split3, split4 = tf.split(output, [32,32,32,32], 3)

        #TODO: Figure out what this function needs to return


    def image_encoder(im):
        '''
        An image encoder, which consists of convolutional layers extracting segments from the input image I
        Our image encoder (Fig. 3c) operates on four different scaled versions of the input image I:
            (256×256, 128×128, 64×64, and 32×32)
        '''
        im_256, im_128, im_64, im_32 = generate_pyramid(im)

    def cross_convolution(x):
        '''
        a cross convolutional layer, which takes the output of the image encoder and the kernel decoder,
        and convolves the image segments with motion kernels
        '''

    def motion_decoder(x):
        '''

        '''


# decoder_hidden = Dense(512, activation='relu')
# decoder_out = Dense(784, activation='sigmoid')

# h_p = decoder_hidden(z)
# outputs = decoder_out(h_p)


# # Overall VAE model, for reconstruction and training
# vae = Model(inputs, outputs)

# # Encoder model, to encode input into latent variable
# # We use the mean as the output as it is the center point, the representative of the gaussian
# encoder = Model(inputs, mu)

# # Generator model, generate new data given latent variable z
# d_in = Input(shape=(n_z,))
# d_h = decoder_hidden(d_in)
# d_out = decoder_out(d_h)
# decoder = Model(d_in, d_out)

# def vae_loss(y_true, y_pred):
#     """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
#     # E[log P(X|z)]
#     recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
#     # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
#     kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

#     return recon + kl


# vae.compile(optimizer='adam', loss=vae_loss)
# vae.fit(X_train, X_train, batch_size=m, nb_epoch=n_epoch)




