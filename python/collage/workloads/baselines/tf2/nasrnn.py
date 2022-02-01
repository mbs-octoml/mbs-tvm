import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul, measure_tf2_gpu

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# hidden_size = 512
hidden_size = 2560
length = 5

def combine(x, h):
    w1 = make_matmul(x, hidden_size)
    w2 = make_matmul(h, hidden_size)
    return tf.add(tf.nn.relu(w1), tf.nn.relu(w2))

def nas_node(input, x):
    t = [combine(x, input) for i in range(8)]
    midt = list()
    midt.append(tf.add(tf.nn.relu(t[0]), tf.nn.sigmoid(t[3])))
    midt.append(tf.add(tf.nn.sigmoid(t[1]), tf.nn.tanh(t[2])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[4]), tf.nn.tanh(t[5])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[6]), tf.nn.relu(t[7])))
    midt.append(tf.add(tf.nn.sigmoid(midt[1]), tf.nn.tanh(midt[2])))
    midt.append(tf.multiply(tf.nn.tanh(midt[0]), tf.nn.tanh(midt[3])))
    midt.append(tf.multiply(tf.nn.tanh(midt[4]), tf.nn.tanh(midt[5])))
    return tf.nn.tanh(midt[6])

def nasrnn_tf2_model(xs):
    input_dictionary = {}
    state = tf.constant(np.random.random_sample((1, hidden_size)), dtype=tf.float32)
    for i in range(length):
        state = nas_node(state, xs)#[i])
    return state

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def nasrnn_tf2(xs):
    return nasrnn_tf2_model(xs)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def nasrnn_tf2_xla(xs):
    return nasrnn_tf2_model(xs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'nasrnn'
    input_shape = (args.batch_size, 2560)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(nasrnn_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(nasrnn_tf2_xla, inputs, method_name, args)
