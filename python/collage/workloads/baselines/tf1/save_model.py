import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul

def attention(input, heads):
    d_model = input.shape[1].value
    q = make_matmul(input, d_model)
    k = make_matmul(input, d_model)
    v = make_matmul(input, d_model)
    # reshape query, key, value
    q = tf.reshape(q, shape=(64,16,64))
    k = tf.reshape(k, shape=(64,16,64))
    v = tf.reshape(v, shape=(64,16,64))
    # transpose q, k, v for batched matmul
    q = tf.transpose(q, perm=(1,0,2))
    k = tf.transpose(k, perm=(1,0,2))
    v = tf.transpose(v, perm=(1,0,2))
    logits = tf.matmul(q, k)
    output = tf.matmul(logits, v)
    # transpose the output back
    output = tf.transpose(output, perm=(1,0,2))
    output = tf.reshape(output, shape=(64, 1024))
    # a final linear layer
    output = make_matmul(tf.nn.relu(make_matmul(input, 4*d_model)), d_model)
    return output

input = tf.placeholder(tf.float32, shape=(64,1024))
input_dictionary = {}
input_dictionary[input] = np.random.random_sample((64, 1024))
t = input
for i in range(12):
    t = attention(t, 16)

output_nodes = [t]

config = tf.ConfigProto()

# Input model name
model_name = "bert"
with tf.Session(config=config) as sess:
    # Saving
    inputs = {"input_placeholder": input}
    outputs = {"prediction": output_nodes[0]}
    tf.saved_model.simple_save(
        sess, f'models_pb/{model_name}', inputs, outputs
    )