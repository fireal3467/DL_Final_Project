import tensorflow as tf

def generate_pyramid(input_im_128):
        im_256 = tf.image.resize_images(input_im_128, tf.constant([256,256]), method=ResizeMethod.BILINEAR)
        im_128 = tf.image.resize_images(input_im_128, tf.constant([128,128]), method=ResizeMethod.BILINEAR)
        im_64 = tf.image.resize_images(input_im_128, tf.constant([64,64]), method=ResizeMethod.BILINEAR)
        im_32 = tf.image.resize_images(input_im_128, tf.constant([32,32]), method=ResizeMethod.BILINEAR)

        return [im_256, im_128, im_64,im_32]

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

def conv2d_layer(inputs, filters, kernel_size=(5,5), strides=(1,1), padding='SAME', activation=tf.nn.relu):
    return tf.layers.batch_normalization(
        tf.layers.conv2d(
            inputs=inputs, 
            filters=filters, 
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        ))

def get_difference_image(cur_im, next_im):
    '''
    notes: possibly make this batchsize by batchsize.
    input: 
    cur_im = the current image of size [128,128,3]
    next_im = the next image of size [128,128,3]

    return:
    a difference image of size [128,128,3]
    '''
    # maybe reference the original image preprocessing code:
    #   https://github.com/tfxue/visual-dynamics/blob/master/src/utilfunc.lua 

    output = tf.subtract(next_im, cur_im)
    #normalization:
    #output = tf.add(tf.div(output,2), 127)
    return output
   

'''

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

'''

