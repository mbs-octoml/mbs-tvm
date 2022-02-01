import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_activation, make_conv2d, make_conv2d_bn, measure_tf2_gpu

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def resnext_block(input, strides, out_channels, groups, name):
    t = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name=name+"_conv1")
    t = tf.split(t, groups, axis=1, name=name+"_split")
    assert(len(t) == groups)
    for i in range(groups):
        t[i] = make_conv2d(input_tensor=t[i], filter_shape=(3,3,t[i].shape[1],out_channels//groups), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv2_".format(i))
    output = tf.concat(t, axis=1, name=name+"_concat")
    t = make_conv2d(input_tensor=output, filter_shape=(1,1,output.shape[1],2*out_channels), strides=(1,1,1,1), padding="SAME", actimode="NONE", name=name+"_conv3")
    if (strides[2]>1) or (input.shape[1] != out_channels*2):
        input = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels*2), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv4")
    return tf.nn.relu(tf.add(input, t))

def resnext50_tf2_model(input):
    t = input
    for i in range(3):
        t = resnext_block(t, (1,1,1,1), 128, 32, "resnet_block_1_{}".format(i))
    strides=(1,1,2,2)
    for i in range(4):
        t = resnext_block(t, strides, 256, 32, "resnet_block_2_{}".format(i))
        strides=(1,1,1,1)
    strides=(1,1,2,2)
    for i in range(6):
        t = resnext_block(t, strides, 512, 32, "resnet_block_3_{}".format(i))
        strides=(1,1,1,1)
    strides=(1,1,2,2)
    for i in range(3):
        t = resnext_block(t, strides, 1024, 32, "resnet_block_4_{}".format(i))
        strides=(1,1,1,1)
    return t

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def resnext50_tf2(input):
    return resnext50_tf2_model(input)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def resnext50_tf2_xla(input):
    return resnext50_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'resnext50_32x4d'
    input_shape = (args.batch_size, 64, 56, 56)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(resnext50_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(resnext50_tf2_xla, inputs, method_name, args)
