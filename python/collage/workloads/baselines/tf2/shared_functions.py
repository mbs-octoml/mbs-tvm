import tensorflow as tf
import numpy as np
import argparse
import time
from tqdm import tqdm

import sys
import os
# Hacky way to include function from parent scripts
this_code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,f'{this_code_path}/../../')

from measure_end_to_end import log_e2e_perf

def make_activation(input, actimode, name):
    if actimode == "NONE":
        return input
    elif actimode == "RELU":
        relu_name = name + "_relu"
        relu = tf.nn.relu(input, name=relu_name)
        return relu
    elif actimode == "SIGMOID":
        sigmoid_name = name + "_sigmoid"
        sigmoid = tf.nn.sigmoid(input, name=sigmoid_name)
        return sigmoid
    elif actimode == "TANH":
        tanh_name = name + "_tanh"
        tanh = tf.nn.tanh(input, name=tanh_name)
        return tanh
    else:
        print("Unknown Actimode")
        assert(0)

def make_conv2d(input_tensor, filter_shape, strides, padding, actimode, name):
    weights_name = name + "_weights"
    conv_name = name + "_conv2d"
    weights = tf.constant(np.random.random_sample(filter_shape), name=weights_name, dtype=tf.float32)
    conv2d = tf.nn.conv2d(input=input_tensor, filters=weights, strides=strides, padding=padding, data_format="NCHW", name=conv_name)
    return make_activation(conv2d, actimode, name)

def make_conv2d_nhwc(input_tensor, filter_shape, strides, padding, actimode, name):
    weights_name = name + "_weights"
    conv_name = name + "_conv2d"
    weights = tf.constant(np.random.random_sample(filter_shape), name=weights_name, dtype=tf.float32)
    conv2d = tf.nn.conv2d(input=input_tensor, filters=weights, strides=strides, padding=padding, data_format="NHWC", name=conv_name)
    return make_activation(conv2d, actimode, name)

def make_conv3d(input_tensor, filter_shape, strides, padding, actimode, name):
    weights_name = name + "_weights"
    conv_name = name + "_conv3d"
    weights = tf.constant(np.random.random_sample(filter_shape), name=weights_name, dtype=tf.float32)
    conv2d = tf.nn.conv3d(input=input_tensor, filters=weights, strides=strides, padding=padding, data_format="NCDHW", name=conv_name)
    return make_activation(conv2d, actimode, name)

def make_conv2d_bn(input_tensor, filter_shape, strides, padding, actimode, name):
    weights_name = name + "_weights"
    conv_name = name + "_conv2d"
    mean_name = name + "_mean"
    variance_name = name + "_var"
    offset_name = name + "_offset"
    scale_name = name + "_scale"
    weights = tf.constant(np.random.random_sample(filter_shape), name=weights_name, dtype=tf.float32)
    conv2d = tf.nn.conv2d(input=input_tensor, filters=weights, strides=strides, padding=padding, data_format="NCHW", name=conv_name)

    mean = tf.constant(np.random.random_sample((1,conv2d.shape[1], 1, 1)), name=mean_name, dtype=tf.float32)
    variance = tf.constant(np.random.random_sample((1,conv2d.shape[1], 1, 1)), name=variance_name, dtype=tf.float32)
    offset = tf.constant(np.random.random_sample((1,conv2d.shape[1], 1, 1)), name=offset_name, dtype=tf.float32)
    scale = tf.constant(np.random.random_sample((1,conv2d.shape[1], 1, 1)), name=scale_name, dtype=tf.float32)
    bn = tf.nn.batch_normalization(conv2d, mean, variance, offset, scale, 1e-7, name= name + "_bn")
    return make_activation(bn, actimode, name)

def make_seperable_conv2d(input_tensor, out_channels, kernels, strides, padding, actimode="NONE", name="seperable_conv2d"):
    depthwise_filter_shape=(kernels[0],kernels[1],input_tensor.shape[1],1)
    pointwise_filter_shape=(1,1,input_tensor.shape[1],out_channels)
    dp_filter = tf.constant(np.random.random_sample(depthwise_filter_shape), name=name+"_dp_filter", dtype=tf.float32)
    pw_filter = tf.constant(np.random.random_sample(pointwise_filter_shape), name=name+"_pw_filter", dtype=tf.float32)
    conv2d = tf.nn.separable_conv2d(input=input_tensor, depthwise_filter=dp_filter, pointwise_filter=pw_filter, strides=strides, padding=padding, data_format="NCHW", name=name)
    return make_activation(conv2d, actimode, name)

def make_avgpool2d(input_tensor, kernels, strides, padding):
    return tf.nn.avg_pool2d(input=input_tensor, ksize=kernels, strides=strides, padding=padding, data_format="NCHW")

def make_maxpool2d(input_tensor, kernels, strides, padding):
    return tf.nn.max_pool2d(input=input_tensor, ksize=kernels, strides=strides, padding=padding, data_format="NCHW")

def make_linear(input_tensor, out_channels):
    weight_shape = (input_tensor.shape[1], out_channels)
    bias_shape = (1, out_channels)
    weight = tf.constant(np.random.random_sample(weight_shape), dtype=tf.float32)
    bias = tf.constant(np.random.random_sample(bias_shape), dtype=tf.float32)
    return tf.math.add(tf.matmul(input_tensor, weight), bias)
 
def make_matmul(input_tensor, out_channels):
    weight_shape = (input_tensor.shape[1], out_channels)
    weight = tf.constant(np.random.random_sample(weight_shape), dtype=tf.float32)
    return tf.matmul(input_tensor, weight)

def tf2_args_checker(args, parser):
    is_missing_arg = not args.network
    is_missing_arg |= not args.hw
    # is_missing_arg |= not args.batch_size
    # is_missing_arg |= not args.target
    # is_missing_arg |= not args.dtype

    if is_missing_arg:
        parser.error('Make sure you input all arguments')

def get_tf2_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")

    # Measurement related parameters
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=50)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=10)
    # parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=10000)
    # parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=2000)
    args = parser.parse_args()

    tf2_args_checker(args, parser)
    return args

# Don't forget to sync this code with
def measure_tf2_gpu(model, inputs, method_name, args):
    # discard_iter, iterations = 2000, 10000
    discard_iter, iterations = 100, 1000
    is_perf_logging = True

    # Enable graph mode instead of eager execution
    # Graph mode performs better, so it is fairer to compare against ours
    tf.config.run_functions_eagerly(False)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Warning(@Soo): It has an issue of executing CPU only for TF2.4.0 or TF2.6.0
    # tf.compat.v1.disable_eager_execution()

    device_name = '/device:GPU:0'

    # Change the device depending on hw
    nvidia_gpus = ['rtx2070', 'rtx3070', 'jetson', 'v100']
    if args.hw == 'xeon':
        device_name = '/device:CPU:0'
    elif args.hw in nvidia_gpus:
        pass
    else:
        Exception(f"Unexpected HW for external compiler op pass: {args.hw}")

    with tf.device(device_name):
        # with tf.device('/device:CPU:0'):
        times = []
        for i in tqdm(range(discard_iter + iterations)):
            t0 = time.time()
            model(inputs)
            t1 = time.time()
            times.append(t1 - t0)

    times = 1000.0 * np.array(times)[discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of {method_name} on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, method_name, mean_perf, std_perf, is_perf_logging)
