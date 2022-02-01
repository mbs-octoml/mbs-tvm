from tvm import relay
# import tensorflow as tf

import os
import onnx
import copy

from .workloads import WORKLOADS_DIC

# NOTE: Make sure that you executed codes in "baselines/pytorch_new" to have the most recent onnx files
def get_network_from_onnx(name, batch_size):
    assert name in WORKLOADS_DIC
    # if batch_size > 1, we need to think about how to take care of bert and nasrnn
    assert batch_size == 1

    this_code_path = os.path.dirname(os.path.abspath(__file__))
    onnx_model = onnx.load(f"{this_code_path}/../baselines/pytorch/models/{name}.onnx")

    # Set the input shape dict
    shape_dict = WORKLOADS_DIC[name][batch_size]
    # We should copy shape_dict because shape_dict will be consumed in from_onnx
    shape_dict_tmp = copy.deepcopy(shape_dict)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict_tmp, freeze_params=True)

    print(f"(Loaded network, Shape dict) = ({name}, {shape_dict})")
    return mod, params, shape_dict, None # we don't need output shape