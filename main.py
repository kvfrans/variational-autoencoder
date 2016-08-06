import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import os
from scipy.misc import imsave as ims

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

n_hidden = 500
n_z = 20
batchsize = 100

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

# encoder
def recognition(input_images):
    with tf.variable_scope("recognition"):
        w1 = tf.get_variable("w1",[784,n_hidden])
        b1 = tf.get_variable("b1",[n_hidden])
        w2 = tf.get_variable("w2",[n_hidden,n_hidden])
        b2 = tf.get_variable("b2",[n_hidden])
        w_mean = tf.get_variable("w_mean",[n_hidden,n_z])
        b_mean = tf.get_variable("b_mean",[n_z])
        w_stddev = tf.get_variable("w_stddev",[n_hidden,n_z])
        b_stddev = tf.get_variable("b_stddev",[n_z])

    h1 = tf.nn.sigmoid(tf.matmul(input_images,w1) + b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1,w2) + b2)
    o_mean = tf.matmul(h2,w_mean) + b_mean
    o_stddev = tf.matmul(h2,w_stddev) + b_stddev
    return o_mean, o_stddev

# decoder
def generation(z):
    with tf.variable_scope("generation"):
        w1 = tf.get_variable("w1",[n_z,n_hidden])
        b1 = tf.get_variable("b1",[n_hidden])
        w2 = tf.get_variable("w2",[n_hidden,n_hidden])
        b2 = tf.get_variable("b2",[n_hidden])
        w_image = tf.get_variable("w_image",[n_hidden,784])
        b_image = tf.get_variable("b_image",[784])

    h1 = tf.nn.sigmoid(tf.matmul(z,w1) + b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1,w2) + b2)
    o_image = tf.nn.sigmoid(tf.matmul(h2,w_image) + b_image)
    return o_image

images = tf.placeholder(tf.float32, [None, 784])
# instead of mapping directly to z, map a gaussian over z, parameterized by mean/stddev
# important: z_stddev contains log(standard_deviation^2).
z_mean, z_stddev = recognition(images)
print z_mean.get_shape()

# unit guassian
samples = tf.random_normal([batchsize,n_z],0,1,dtype=tf.float32)
guessed_z = z_mean + (z_stddev * samples)

generated_images = generation(guessed_z)

# -log of p(x|z)
generation_loss = -tf.reduce_sum(images * tf.log(1e-10 + generated_images) + (1-images) * tf.log(1e-10 + 1 - generated_images),1)

# we want real p(z) to be unit guassian
# the KL divergence loss between real p(z) and q(z|x)
# q(z|x) is the recognition network
# latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_stddev) - z_stddev - 1,1)
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)

cost = tf.reduce_mean(generation_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

train = True
visualization = mnist.train.next_batch(batchsize)[0]
reshaped_vis = visualization.reshape(batchsize,28,28)
ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
# train
saver = tf.train.Saver(max_to_keep=2)
with tf.Session() as sess:
    if train:
        sess.run(tf.initialize_all_variables())
        for epoch in range(10):
            for idx in range(int(n_samples / batchsize)):
                batch = mnist.train.next_batch(batchsize)[0]
                _, real_cost = sess.run((optimizer, cost), feed_dict={images: batch})
                # dumb hack to print cost every epoch
                if idx % (n_samples - 3) == 0:
                    print "%d: %f" % (epoch, real_cost)
                    saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                    generated_test = sess.run(generated_images, feed_dict={images: visualization})
                    generated_test = generated_test.reshape(batchsize,28,28)
                    ims("results/"+str(epoch)+".jpg",merge(generated_test[:64],[8,8]))
    else:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"/training/"))
        batch = mnist.train.next_batch(batchsize)[0]
