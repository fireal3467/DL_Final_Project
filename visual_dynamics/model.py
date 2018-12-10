import tensorflow as tf
ResizeMethod = tf.image.ResizeMethod
from helpers import *

# NOTE 60k pairs of frames used for training
class Model:
    def __init__(self, cur_im_batch, next_im_batch, latent_size):
        self.orig_image_128 = cur_im_batch
        self.target_image = next_im_batch

        self.difference_image = get_difference_image(cur_im_batch, next_im_batch)

        self.learning_rate = 0.0001
        self.latent_size = latent_size
        self.batch_size = self.orig_image_128.get_shape().as_list()[0]

        self.mu, self.log_sigma = self.motion_encoder()
        self.z = self.sample()

        self.motion_kernels = self.kernel_decoder()
        self.segmented_images = self.image_encoder()
        self.cross_conv_output = self.cross_convolution()
        self.generated_image = self.motion_decoder()

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

        num_channels = int(self.latent_size / 25) # should be 128
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
        return im_block_64, im_block_32, im_block_16, im_block_8

    def cross_convolution(self):
        '''
        A cross convolutional layer takes the output of the image encoder and the kernel decoder,
            and convolves the image segments with motion kernels.
        The cross convolutional layer has the same output size as the image encoder.
        '''

        def apply_cross_conv_to_block(im_block, kernel_block):
            n_channels = im_block.shape[3] # 32
            outputs = []

            # print("kernel_block", kernel_block.shape)
            # print("im_block", im_block.shape)

            for batch_idx in range(self.batch_size):
                this_batch = []

                for channel_idx in range(n_channels):
                    im_size = im_block.shape[1]
                    im = tf.reshape(im_block[batch_idx,:,:,channel_idx], [1, im_size, im_size, 1])
                    kernel = tf.reshape(kernel_block[batch_idx,:,:,channel_idx], [5, 5, 1, 1])

                    single_channel_conv = tf.nn.conv2d(im, kernel, [1,1,1,1], "SAME")
                    this_batch.append(tf.reshape(single_channel_conv, [im_size, im_size]))

                outputs.append(tf.stack(this_batch, axis=2))
            return tf.stack(outputs)

        im_block_64, im_block_32, im_block_16, im_block_8 = self.segmented_images
        k_block_1, k_block_2, k_block_3, k_block_4 = self.motion_kernels

        conv_block_64 = apply_cross_conv_to_block(im_block_64, k_block_1)
        conv_block_32 = apply_cross_conv_to_block(im_block_32, k_block_2)
        conv_block_16 = apply_cross_conv_to_block(im_block_16, k_block_3)
        conv_block_8 = apply_cross_conv_to_block(im_block_8, k_block_4)

        return conv_block_64, conv_block_32, conv_block_16, conv_block_8


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
        output = conv2d_layer(combined_blocks, filters=128, kernel_size=9)
        output = conv2d_layer(combined_blocks, filters=128, kernel_size=1)
        output = conv2d_layer(combined_blocks, filters=3, kernel_size=1)
        return output

    def loss(self):
        '''
        Objective function:
            𝐷𝐾𝐿(𝑞𝜙(𝑧|𝑣𝑠𝑦𝑛,𝐼)||𝑁(𝟎,𝐈)) + 𝜆 ⋅ ||𝑣𝑠𝑦𝑛 − 𝑣𝑔||
        '''

        lambda_term = 1E-5 # Hyperparameter?
        log_squared_sigma = 2 * self.log_sigma

        # reduce along the first
        self.kl_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.exp(log_squared_sigma) - log_squared_sigma - 1.0)
        self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(self.generated_image, self.difference_image))

        return self.kl_divergence + lambda_term * self.reconstruction_loss

    def optimizer(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return opt.minimize(self.loss_value)
