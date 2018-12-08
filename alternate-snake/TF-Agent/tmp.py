class Model:

    def generate_pyramid(im):
        pass 


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

