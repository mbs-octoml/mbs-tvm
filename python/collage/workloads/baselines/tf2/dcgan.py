import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_activation, make_conv2d_bn, make_linear, measure_tf2_gpu
from shared_functions import make_conv2d_nhwc as make_conv2d

NAME = 'dcgan'
batch_size = 1
latent_dim = 100
img_size = 256
channels = 3

def generator(input):
    t = input
    init_size = img_size // 4
    l1 = make_linear(t, 128 * init_size ** 2)
    #print(l1.shape)

    t = tf.reshape(l1, (l1.shape[0], init_size, init_size, 128) )

    #print(t.shape)

    new_height = int(round(init_size * 2))
    new_width = int(round(init_size * 2))
    resized = tf.image.resize(t, [new_height, new_width])

    #print(resized.shape)

    t = make_conv2d(input_tensor=resized, filter_shape=(3,3,128,128), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)

    new_height = int(round(init_size * 2 * 2))
    new_width = int(round(init_size * 2 * 2))
    resized = tf.image.resize(t, [new_height, new_width])
    t = make_conv2d(input_tensor=resized, filter_shape=(3,3,128,64), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,64,channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.tanh(t)
    return t

def discriminator(input):
    t = input
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,channels,16), strides=2, padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,16,32), strides=2, padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,32,64), strides=2, padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,64,128), strides=2, padding="SAME", actimode="RELU", name="conv")
    t = tf.nn.relu(t)

    ds_size = img_size // 2 ** 4
    t = tf.reshape(t, (t.shape[0], 128 * ds_size ** 2))
    adv_layer = make_linear(t, 1)
    t = tf.nn.sigmoid(adv_layer)
    return t

def dcgan_tf2_model(input):
    t = generator(input)
    t = discriminator(t)
    return t

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def dcgan_tf2(input):
    return dcgan_tf2_model(input)

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=True)
def dcgan_tf2_xla(input):
    return dcgan_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'dcgan'
    input_shape = (args.batch_size, 100)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(dcgan_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(dcgan_tf2_xla, inputs, method_name, args)
