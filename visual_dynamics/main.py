import tensorflow as tf
import numpy as np
import os
from model import Model
import scipy.misc

import png

BATCH_SIZE = 4
LATENT_SIZE = 3200
NUM_EPOCHS = 60


output_directory = "/research/xai/starcraft/users/bweissm1/"
# output_directory = os.path.expanduser("~/Desktop/model_out/")
image_directory = "../3Shapes2_large/"
print_every = 20
save_every = 100

#------------------------------------------------------------------------------------------------------
#taken from cs1470 hw8
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dirname, batch_size=128,
    shuffle_buffer_size=250000, n_threads=2):

    root_dir = os.path.expanduser(dirname)
    filenames = [f"{root_dir}train.txt"]

    def load_im(image_id, next_or_cur):
        filename = tf.strings.join([tf.constant(root_dir), image_id, tf.constant(f"_im{next_or_cur}.png")])
        image = tf.image.decode_png(tf.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, tf.constant([128,128]))
        return image

    def load_images_from_id(index):
        return load_im(index,1), load_im(index,2)

    # List filenames
    dataset = tf.data.TextLineDataset(filenames)

    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_images_from_id, num_parallel_calls=n_threads)
    # Create batch, dropping the final one which has less than batch_size elements
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset that we can
    #    re-initialize for each new epoch
    return dataset.make_initializable_iterator()

#------------------------------------------------------------------------------------------------------
#Set up data collection, initialize the model

print("Welcome!")

print(f"Using output directory: {output_directory}")
if not os.path.isdir(output_directory):
    print ("Output directory does not exist!")
    exit(1)

dataset_iterator = load_image_batch(image_directory,
    batch_size=BATCH_SIZE)

print(f"Dataset successfully defined! Using root directory: {image_directory}")

cur_images_batch, next_images_batch = dataset_iterator.get_next()

model = Model(cur_images_batch, next_images_batch, LATENT_SIZE)

print("Model successfully initialized!")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#------------------------------------------------------------------------------------------------------
#Set up trainer

def train():
    print("Entering training loop!")
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print('========================== EPOCH %d  ==========================' % epoch)

        # Initialize the dataset iterator for this epoch (this shuffles the
        #    data differently every time)
        print("Attempting to initialize the dataset...")
        sess.run(dataset_iterator.initializer)
        print("Dataset initialized!")

        # Loop over our data until we run out
        losses = []
        iteration = 0
        try:
            while True:
                loss, _, kl_loss, recon_loss = sess.run([
                    model.loss_value,
                    model.train_op,
                    model.kl_divergence,
                    model.reconstruction_loss
                ], feed_dict={model.is_training: True})

                losses.append((loss, kl_loss, recon_loss))

                # Print losses
                if iteration % print_every  == 0:
                    slice_losses = losses[-10:]
                    n = float(len(slice_losses))
                    avg_loss = sum(x[0] for x in slice_losses) / n
                    avg_kl = sum(x[1] for x in slice_losses) / n
                    avg_recon = sum(x[2] for x in slice_losses) / n
                    print('(%d) Iteration %d: loss = %g \t kl_loss = %g \t recon_loss = %g' % (epoch, iteration, avg_loss, avg_kl, avg_recon))
                
                if iteration % save_every == 0:
                    print("Iteration finished! Saving model")
                    saver.save(sess, f'{output_directory}saved_model')

                    test(f"e{epoch}i{iteration}")

                    with open(f"{output_directory}log.txt", 'w+') as f:
                        f.write(f"Epoch: {epoch} Iteration: {iteration}")

                iteration += 1
        except tf.errors.OutOfRangeError:
            # Triggered when the iterator runs out of data
            pass

        # Save at the end of the epoch, too
        saver.save(sess, f'{output_directory}saved_model')


def test(identifier):
    print("Generating test images...")

    loss, kl_loss, recon_loss, real_diff, gen_diff = sess.run([
        model.loss_value,
        model.kl_divergence,
        model.reconstruction_loss,
        model.difference_image, 
        model.generated_image
    ], feed_dict={model.is_training: False})

    print("Saving images...")
    print(f"Loss: {loss}\t kl: {kl_loss}\t recon: {recon_loss}")

    for i in range(real_diff.shape[0]):
        out = np.concatenate((real_diff[i,:,:,:], gen_diff[i,:,:,:]))
        scipy.misc.imsave(f"{output_directory}img_{identifier}_{i}.jpg", out)

#------------------------------------------------------------------------------------------------------
#Run everything

train()
