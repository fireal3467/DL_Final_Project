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


    def upsample_block(im_block, scale=scale):
        pass
        # NOTE 60k pairs of frames used for training

    def image_encoder(orig_image_128):
        '''
        An image encoder, which consists of convolutional layers extracting segments from the input image I
        Our image encoder (Fig. 3c) operates on four different scaled versions of the input image I:
            (256×256, 128×128, 64×64, and 32×32)
        '''
        im_256, im_128, im_64, im_32 = generate_pyramid(im)

        im_block_64 = im_encode_convolutions(im_256)
        im_block_32 = im_encode_convolutions(im_128)
        im_block_16 = im_encode_convolutions(im_64)
        im_block_8 = im_encode_convolutions(im_32)

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


    def cross_convolution(image_block, motion_kernels):
        '''
        A cross convolutional layer takes the output of the image encoder and the kernel decoder, 
            and convolves the image segments with motion kernels.
        The cross convolutional layer has the same output size as the image encoder.
        '''

        n_channels = image_block.shape[3] # 32
        outputs = []

        for i in range(n_channels):
            im = image_block[:,:,:,i]
            kernel = motion_kernels[:,:,:,i]

            output.append(tf.nn.conv2d(im, kernel, [1,1,1,1], "SAME"))

        return tf.stack(output)


    def motion_decoder(im_block_64, im_block_32, im_block_16, im_block_8):
        '''
        A motion decoder, which regresses the difference image from the combined feature maps.
        '''

        # Our motion decoder (Fig. 3e) starts with an up-sampling layer at each scale, 
        #   making the output of all scales of the cross convolutional layer have a resolution of 64×64. 
        upsampled_64 = upsample_block(im_block_64, scale=2)
        upsampled_32 = upsample_block(im_block_32, scale=4)
        upsampled_16 = upsample_block(im_block_16, scale=8)
        upsampled_8 = upsample_block(im_block_8, scale=16)

        # THIS IS INCORRECT ^^ THESE ARE NOW 128x128 not 64x64

        combined_blocks = tf.concat([upsampled_64, upsampled_32, upsampled_16, upsampled_8], 3)

        # This is then followed by one 9×9 and two 1×1 convolutional and batch normalization layers,
        #     with {128, 128, 3} channels.
        output = tf.layers.batch_normalization(tf.layers.conv2d(combined_blocks, filters=128, kernel_size=9, activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=128, kernel_size=1, activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=3, kernel_size=1, activation=tf.nn.relu))

        return output





