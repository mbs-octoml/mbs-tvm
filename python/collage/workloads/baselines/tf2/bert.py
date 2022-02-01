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

def attention(input, heads):
    d_model = input.shape[1]
    d_k = d_model // heads
    assert d_model % heads == 0
    q = make_matmul(input, d_model)
    k = make_matmul(input, d_model)
    v = make_matmul(input, d_model)
    # reshape query, key, value
    q = tf.reshape(q, shape=(64,16,64))
    k = tf.reshape(k, shape=(64,16,64))
    v = tf.reshape(v, shape=(64,16,64))
    # transpose q, k, v for batched matmul
    q = tf.transpose(a=q, perm=(1,0,2))
    k = tf.transpose(a=k, perm=(1,0,2))
    v = tf.transpose(a=v, perm=(1,0,2))

    logits = tf.matmul(q, k)
    output = tf.matmul(logits, v)
    # transpose the output back
    output = tf.transpose(a=output, perm=(1,0,2))
    output = tf.reshape(output, shape=(64, 1024))
    # a final linear layer
    output = make_matmul(tf.nn.relu(make_matmul(input, 4*d_model)), d_model)
    return output

def bert_tf2_model(input):
    t = input
    for i in range(8):
        t = attention(t, 16)
    return t

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def bert_tf2(input):
    return bert_tf2_model(input)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def bert_tf2_xla(input):
    return bert_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'bert'
    input_shape = (64, 1024)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(bert_tf2, inputs, method_name, args)

    method_name = 'TF-XLA'
    measure_tf2_gpu(bert_tf2_xla, inputs, method_name, args)
