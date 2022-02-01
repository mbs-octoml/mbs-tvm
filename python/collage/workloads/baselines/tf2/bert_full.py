import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul, measure_tf2_gpu
import os

this_code_path = os.path.dirname(os.path.abspath(__file__))
model_path = f"{this_code_path}/../onnx/bert_full.pb"
model = tf.saved_model.load(model_path)

def bert_full_tf2_model(input):
    return model(input0=input)

# @tf.function(jit_compile=False)
@tf.function(experimental_compile=False)
def bert_full_tf2(input):
    return bert_full_tf2_model(input)

# @tf.function(jit_compile=True)
@tf.function(experimental_compile=True)
def bert_full_tf2_xla(input):
    return bert_full_tf2_model(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1, type=int, help="batch size")
    args = parser.parse_args()

    args.network = 'bert_full'
    input_shape = (args.batch_size, 64, 256)
    inputs = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    method_name = 'TF'
    measure_tf2_gpu(bert_full_tf2, inputs, method_name, args)

    method_name = 'TF-XLA'
    measure_tf2_gpu(bert_full_tf2_xla, inputs, method_name, args)
