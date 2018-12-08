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

        self.z = motion_encoder(inputs)
        self.kernels = kernel_decoder(z)

    def motion_encoder(x, difference_image):
        '''
        Input: 
        x - current image with size [batch_size , 128, 128, 3]
        difference_image - the difference image between the current image and the next image with size [batch_size, 128,128,3]
        
        output: [Batch Size, N_Z]
        '''
        #TODO: 
        #add some relu or batch normalizing in between
        #figure out what they mean by a 5x5 conv
        

        #concating them together to get a [batch_size, 128,128,6] image
        output = tf.concat([x,difference_image], 3)



        output = tf.layers.conv2d(x, filters=96,  kernel_size=(5,5))


        output = tf.layers.conv2d(output, filters=96,  kernel_size=(5,5))
        output = tf.layers.batch_normalization(output)
        output = tf.layers.maxpooling(output, pool_size=[1,2,2,1], strides=[1,2,2,1])

        output = tf.layers.conv2d(output, filters=126,  kernel_size=(5,5))
        output = tf.layers.batch_normalization(output)
        output = tf.layers.maxpooling(output, pool_size=[1,2,2,1], strides=[1,2,2,1])

        output = tf.layers.conv2d(output, filters=126,  kernel_size=(5,5))
        output = tf.layers.batch_normalization(output)
        output = tf.layers.maxpooling(output, pool_size=[1,2,2,1], strides=[1,2,2,1])

        output = tf.layers.conv2d(output, filters=256,  kernel_size=(5,5))
        output = tf.layers.batch_normalization(output)
  
        output = tf.layers.conv2d(output, filters=256,  kernel_size=(5,5))
        output = tf.layers.batch_normalization(output)
        output = tf.layers.maxpooling(output, pool_size=[1,4,4,1], strides=[1,3,3,1])

        #what is the shape at the end of this? 256,5,5. The thing paper says it should be 256,5,5?
        reshaped = tf.reshape(output, [BATCH_SIZE, 6400]) 

        #Split the thing into two [batchsize, 3200] vectors to get a mean and variance vector
        mean_vector, variance_vector = tf.split(reshaped, [3200, 3200], 1)
        mu = tf.layers.dense(mean_vector, N_Z)
        #Q: Why is this log_sigma?
        log_sigma = tf.layers.dense(mean_vector, N_Z)


        #take a sample from random normal here
        eps = tf.random_normal(shape=[BATCH_SIZE, N_Z], mean =0., std=1.)

        return mu + tf.exp(log_sigma/2) * eps

    def kernel_decoder(z):
        '''
        input: z of size [BATCH_SIZE, 3200]
        '''

        output = tf.reshape(z, [BATCH_SIZE,5,5,128])
        #Q: Honestly, not sure if this shape will work out rn
        output = tf.layers.conv2d(output, filters=128, kernel_size=(5,5), padding='SAME')
        output = tf.layers.conv2d(output, filters=128, kernel_size=(5,5), padding='SAME')
        split1, split2, split3, split4 = tf.split(output, [32,32,32,32], 3)

        #TODO: Figure out what this function needs to return








decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(X_train, X_train, batch_size=m, nb_epoch=n_epoch)




