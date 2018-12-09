import tensorflow as tf
layers = tf.layers
import numpy as np

from helpers import conv2d_layer

BATCH_SIZE = 50

# NOTE 60k pairs of frames used for training
class Model:
    def __init__(self, cur_im, next_im, batch_size, latent_size):
        self.orig_image_128 = cur_im
        self.target_image = next_im

        self.difference_image = get_difference_image(cur_im, next_im)

        self.latent_size = latent_size
        self.batch_size = batch_size

        self.mu, self.log_sigma = self.motion_encoder()
        self.z = self.sample()

        self.motion_kernels = self.kernel_decoder()
        self.segmented_images = self.image_encoder()
        self.cross_conv_output = self.cross_convolution()
        self.generated_image = self.motion_encoder()

        self.loss_value = self.loss()
        self.train_op = self.optimizer()

    def motion_encoder(self):
        '''
        Input: 
        x - current image with size [batch_size , 128, 128, 3]
        difference_image - the difference image between the current image and the next image with size [batch_size, 128,128,3]
        
        output: [Batch Size, N_Z]
        '''

        # concating them together to get a [batch_size, 128,128,6] image
        output = tf.concat([self.orig_image_128, self.difference_image], 3)

        #[batch_size, 128,128,6]

        output = conv2d_layer(output, filters=96)
        output = conv2d_layer(output, filters=96)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2) # QUESTION: should we use padding: same?
        
        #[batch_size, 64,64,96]

        output = conv2d_layer(output, filters=128)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
        
        # [batch_size, 32, 32, 128]

        output = conv2d_layer(output, filters=128)
        output = conv2d_layer(output, filters=256)
        output = tf.layers.max_pooling2d(output, pool_size=2, strides=2)
  
        # [batch_size, 16,16, 256]

        output = conv2d_layer(output, filters=256)
        output = tf.layers.max_pooling2d(output, pool_size=4, strides=3)
        
        # [batch_size, 5, 5, 256]

        reshaped = tf.reshape(output, [-1, 6400]) 
        mean_vector, variance_vector = tf.split(reshaped, [3200, 3200], 1)

        # Q: Why is this log_sigma? A: https://www.reddit.com/r/MachineLearning/comments/74dx67/d_why_use_exponential_term_rather_than_log_term/
        # Basically, it makes sigma strictly positive to work in log space
        log_sigma = tf.layers.dense(mean_vector, self.latent_size)
        mu = tf.layers.dense(mean_vector, self.latent_size)

        return mu, log_sigma

    def sample(self):

        #take a sample from random normal here
        eps = tf.random_normal(shape=[self.batch_size, self.latent_size])
        return self.mu + tf.exp(0.5 * self.log_sigma) * eps

    def kernel_decoder(self):
        '''
        input: z of size [BATCH_SIZE, N_Z]
        '''

        num_channels = self.latent_size / 25 # should be 128
        output = tf.reshape(self.z, [self.batch_size, 5, 5, num_channels])

        # size [_, 5, 5, 128]

        output = conv2d_layer(output, filters=128)
        output = conv2d_layer(output, filters=128)

        return tf.split(output, num_or_size_splits=4, axis=3)

    def image_encoder(self):
        '''
        An image encoder, which consists of convolutional layers extracting segments from the input image I
        Our image encoder (Fig. 3c) operates on four different scaled versions of the input image I:
            (256×256, 128×128, 64×64, and 32×32)
        '''
        im_256, im_128, im_64, im_32 = generate_pyramid(self.orig_image_128)

        im_block_64 = im_encode_convolutions(im_256)
        im_block_32 = im_encode_convolutions(im_128)
        im_block_16 = im_encode_convolutions(im_64)
        im_block_8 = im_encode_convolutions(im_32)

        # Therefore, the output size of the four channels are 32×64×64, 32×32×32, 32×16×16, and 32×8×8, respectively.
        # In order: [batch_size, 64,64,32], [batch_size, 32, 32,32], [batch_size,16,16,32], [batch_size,8,8,32]  
        return conv_256, conv_128, conv_64, conv_32

    def cross_convolution(self):
        '''
        A cross convolutional layer takes the output of the image encoder and the kernel decoder, 
            and convolves the image segments with motion kernels.
        The cross convolutional layer has the same output size as the image encoder.
        '''

        n_channels = self.segmented_images.shape[3] # 32
        outputs = []

        for i in range(n_channels):
            im = self.segmented_images[:,:,:,i]
            kernel = self.motion_kernels[:,:,:,i]

            output.append(tf.nn.conv2d(im, kernel, [1,1,1,1], "SAME"))

        return tf.stack(output)

    def motion_decoder(self):
        '''
        A motion decoder, which regresses the difference image from the combined feature maps.
        '''

        im_block_64, im_block_32, im_block_16, im_block_8 = self.cross_conv_output

        # Our motion decoder (Fig. 3e) starts with an up-sampling layer at each scale, 
        #   making the output of all scales of the cross convolutional layer have a resolution of 64×64. 
        upsampled_64 = tf.image.resize_images(im_block_64, tf.constant([128,128]), method=ResizeMethod.BILINEAR)
        upsampled_32 = tf.image.resize_images(im_block_32, tf.constant([128,128]), method=ResizeMethod.BILINEAR)
        upsampled_16 = tf.image.resize_images(im_block_16, tf.constant([128,128]), method=ResizeMethod.BILINEAR)
        upsampled_8 = tf.image.resize_images(im_block_8, tf.constant([128,128]), method=ResizeMethod.BILINEAR)

        # THIS IS INCORRECT ^^ THESE ARE NOW 128x128 not 64x64

        combined_blocks = tf.concat([upsampled_64, upsampled_32, upsampled_16, upsampled_8], 3)

        # This is then followed by one 9×9 and two 1×1 convolutional and batch normalization layers,
        #     with {128, 128, 3} channels.
        output = tf.layers.batch_normalization(tf.layers.conv2d(combined_blocks, filters=128, kernel_size=9, activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=128, kernel_size=1, activation=tf.nn.relu))
        output = tf.layers.batch_normalization(tf.layers.conv2d(output, filters=3, kernel_size=1, activation=tf.nn.relu))

        return output

    def loss(self):
        '''
        Objective function: 
            𝐷𝐾𝐿(𝑞𝜙(𝑧|𝑣𝑠𝑦𝑛,𝐼)||𝑁(𝟎,𝐈)) + 𝜆 ⋅ ||𝑣𝑠𝑦𝑛 − 𝑣𝑔||
        '''

        lambda_term = 1.0 # Hyperparameter?
        log_squared_sigma = 2 * log_sigma

        # reduce along the first 
        kl_divergence = 0.5 * tf.sum(tf.square(self.mu) + tf.exp(log_squared_sigma) - log_squared_sigma - 1.0)
        reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.generated_image, self.target_image))

        return kl_divergence + lambda_term * reconstruction_loss 

    def optimizer(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return opt.minimize(self.loss_value)


