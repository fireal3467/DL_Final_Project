import tensorflow as tf
layers = tf.layers
image = tf.image
import numpy as np
from scipy.misc import imsave
import os
import argparse


class Model:

    def generate_pyramid(im):
        im_256 = tf.image.resize_images(im, tf.constant([256,256]), method=ResizeMethod.BILINEAR)
        im_128 = tf.image.resize_images(im, tf.constant([128,128]), method=ResizeMethod.BILINEAR)
        im_64 = tf.image.resize_images(im, tf.constant([64,64]), method=ResizeMethod.BILINEAR)
        im_32 = tf.image.resize_images(im, tf.constant([32,32]), method=ResizeMethod.BILINEAR)

        return [im_256, im_128, im_64,im_32]


    def image_encoder(orig_image_128):
        '''
        An image encoder, which consists of convolutional layers extracting segments from the input image I
        Our image encoder (Fig. 3c) operates on four different scaled versions of the input image I:
            (256×256, 128×128, 64×64, and 32×32)
        '''
        im_256, im_128, im_64, im_32 = generate_pyramid(im)

        conv_256 = im_encode_convolutions(im_256)
        conv_128 = im_encode_convolutions(im_128)
        conv_64 = im_encode_convolutions(im_64)
        conv_32 = im_encode_convolutions(im_32)

        # Therefore, the output size of the four channels are 32×64×64, 32×32×32, 32×16×16, and 32×8×8, respectively.
        # In order:
        # [batch_size, 64,64,32], [batch_size, 32, 32,32], [batch_size,16,16,32], [batch_size,8,8,32]  
        return [conv_256, conv_128, conv_64, conv_32]


    def im_encode_convolutions(im):
        '''
        four convolutions (5x5) and batch normalizations channels (64, 64, 64, 32)
            two of which are followed by a 2×2 max pooling layer.
        '''

        output = tf.layers.batch_normalization(tf.layers.conv2d(im, filters=64, kernel_size=(5,5), activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=64, kernel_size=(5,5), activation=tf.nn.relu))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        #[_, s/2, s/4, 64]
        
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=64, kernel_size=(5,5), activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=32, kernel_size=(5,5), activation=tf.nn.relu))
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        #[_, s/4, s/4, 32]

        return output


    def cross_convolution(image_segments, motion_kernels):
        '''
        A cross convolutional layer takes the output of the image encoder and the kernel decoder, 
            and convolves the image segments with motion kernels.
        The cross convolutional layer has the same output size as the image encoder.
        '''


    def motion_decoder(x):
        '''

        '''

