import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
from shared_functions import make_activation, make_conv2d, make_conv2d_bn, measure_tf2_gpu

inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

# tf.config.run_functions_eagerly(False)
#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

def block(tensor, inp, oup, stride, expand_ratio):
    convd = tensor

    hidden_dim = int(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == oup

    #print(inp)
    if expand_ratio == 1:
        convd = make_conv2d(input_tensor=convd, filter_shape=(3,3,hidden_dim, hidden_dim), strides=(1,1,stride,stride), padding=[[0, 0], [0, 0],[1, 1], [1, 1]], actimode="RELU", name="conv1")
        convd = make_conv2d(input_tensor=convd, filter_shape=(1,1,hidden_dim, oup), strides=(1,1,1,1), padding=[[0, 0], [0, 0],[0, 0], [0, 0]], actimode="NONE", name="conv1")

        """
        conv = nn.Sequential(
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        )
        """
        tensor = convd
    else:
        convd = make_conv2d(input_tensor=convd, filter_shape=(1,1,inp, hidden_dim), strides=(1,1,1,1), padding=[[0, 0], [0, 0],[1, 1], [1, 1]], actimode="RELU", name="conv1")
        groups = hidden_dim
        t = tf.split(convd, groups, axis=1, name="split")
        assert(len(t) == groups)
        for i in range(groups):
            t[i] = make_conv2d(input_tensor=t[i], filter_shape=(3,3,t[i].shape[1],hidden_dim//groups), strides=(1,1,stride,stride), padding=[[0, 0], [0, 0],[1, 1], [1, 1]], actimode="RELU", name="conv2_".format(i))
        output = tf.concat(t, axis=1, name="concat")
        #convd = make_conv2d(input_tensor=output, filter_shape=(3,3,hidden_dim, hidden_dim), strides=(1,1,stride,stride), padding=[[0, 0], [0, 0],[1, 1], [1, 1]], actimode="RELU", name="_conv1")
        convd = make_conv2d(input_tensor=convd, filter_shape=(1,1,hidden_dim, oup), strides=(1,1,1,1), padding=[[0, 0], [0, 0],[0, 0], [0, 0]], actimode="NONE", name="conv1")

        """
        conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        )
        """
        tensor = convd

    if use_res_connect:
        tensor = tensor + convd

    return tensor

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

def mobilenetv2_tf2_model(input):
    tensor = input
    input_channel = 32
    for t, c, n, s in inverted_residual_setting:
        output_channel = make_divisible(c * 1) if t > 1 else c
        for i in range(n):
            if i == 0:
                tensor = block(tensor, input_channel, output_channel, s, expand_ratio=t)
            else:
                tensor = block(tensor, input_channel, output_channel, 1, expand_ratio=t)
            input_channel = output_channel
    tensor = make_conv2d(input_tensor=tensor, filter_shape=(1,1,input_channel,1280), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="last_conv")
    return tensor

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def mobilenetv2_tf2(input):
    return mobilenetv2_tf2_model(input)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def mobilenetv2_tf2_xla(input):
    return mobilenetv2_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'mobilenet_v2'
    input_shape = (args.batch_size, 32, 224, 224)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(mobilenetv2_tf2, inputs, method_name, args)

    # This errors out; resize kernel is not supported even by the most recent XLA
    method_name = 'TF-XLA'
    measure_tf2_gpu(mobilenetv2_tf2_xla, inputs, method_name, args)
