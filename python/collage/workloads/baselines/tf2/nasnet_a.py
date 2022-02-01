import argparse
import tensorflow as tf
import numpy as np
import time
import torch
# import pretrainedmodels
# import tensorflow_hub as hub
from shared_functions import make_activation, make_conv2d, make_seperable_conv2d, make_avgpool2d, make_maxpool2d, measure_tf2_gpu

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def squeeze(out_channels, input):
    return make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1],out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="squeeze")

def fit(current, input):
    return squeeze(current.shape[1], input)

def normal_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(cur)
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(prev)
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    assert len(ts) == 10
    outputs=list()
    for i in range(5):
        outputs.append(tf.add(ts[2*i], ts[2*i+1]))
    return tf.concat(outputs, axis=1, name="concat1")

def reduction_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    outputs = list()
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[0], ts[1]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[2], ts[3]))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[4], ts[5]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=outputs[0], out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    outputs.append(tf.add(ts[6], ts[7]))
    ts.append(make_avgpool2d(input_tensor=outputs[0], kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(tf.add(ts[8], ts[9]))
    return tf.concat(outputs, axis=1, name="concat2")

# parser = argparse.ArgumentParser()
# parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
# parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
# parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=200)
# parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
# args = parser.parse_args()

def nasneta_tf2_model(input0):
    input = input0
    out_channels = 64
    for i in range(3):
        if i > 0:
            input = reduction_cell(prev, cur, out_channels)
        prev = input
        cur = input
        for j in range(5):
            t = normal_cell(prev, cur, out_channels)
            prev = cur
            cur = t
        out_channels *= 2
    return cur


# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def nasneta_tf2(input0):
    return nasneta_tf2_model(input0)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def nasneta_tf2_xla(input0):
    return nasneta_tf2_model(input0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'nasneta'
    input_shape = (args.batch_size, 64, 56, 56)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(nasneta_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(nasneta_tf2_xla, inputs, method_name, args)
