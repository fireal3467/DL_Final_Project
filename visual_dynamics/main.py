import tf
import numpy as np

BATCH_SIZE = 50
LATENT_SIZE = 3200
NUM_EPOCHS = 10 #TODO: figure this out

#------------------------------------------------------------------------------------------------------
#taken from cs1470 hw8
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))


#TODO: Change this function to load images
def load_image_batch(dirname, batch_size=128,
    shuffle_buffer_size=250000, n_threads=2):
    '''
    return: 
    a dataset of shape [num batches, batch_size, 2, 128,128,3]

    each element in the batch is a tuple of (current image, next image). 
    '''


    # Function used to load and pre-process image files
    # (Have to define this ahead of time b/c Python does allow multi-line
    #    lambdas, *grumble*)
    def load_and_process_image(filename):
        # Load image
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    #TODO: figure out what the current Data representation is and how to access / shuffle it. 


#------------------------------------------------------------------------------------------------------
#Set up data collection, initialize the model

dataset = load_image_batch(args.img_dir,
    batch_size=BATCH_SIZE)

current_images = tf.placeholder(dtype=tf.uint8, shape=[None, 128, 128, 3])
next_images = tf.placeholder(dtype=tf.uint8, shape=[None, 128, 128, 3])

model = Model(current_images, next_images, BATCH_SIZE, LATENT_SIZE)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


#------------------------------------------------------------------------------------------------------
#Set up trainer

def train():
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print('========================== EPOCH %d  ==========================' % epoch)

        # Initialize the dataset iterator for this epoch (this shuffles the
        #    data differently every time)
        sess.run(dataset_iterator.initializer)

        # Loop over our data until we run out
        iteration = 0
        try:
            while True:
                #TODO: finish setting this up after data has been figured out. 
                loss, _ = sess.run([model.loss_value, model.train_op], feed_dict={current_images: , next_images: })

                # Print losses
                if iteration % 10  == 0:
                    print('Iteration %d: loss = %g' % (iteration, loss))
                if iteration % 100 == 0:
                    saver.save(sess, './snake_saved_model')
                iteration += 1
        except tf.errors.OutOfRangeError:
            # Triggered when the iterator runs out of data
            pass

        # Save at the end of the epoch, too
        saver.save(sess, './snake_saved_model')

#------------------------------------------------------------------------------------------------------
#Run everything

train()

