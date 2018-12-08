class Model:

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

