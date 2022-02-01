from tvm import relay

import os
import onnx
import copy
import logging

from .utils import WORKLOADS_DIC

# NOTE: Make sure that you executed codes in "baselines/pytorch_new" to have the most recent onnx files


def get_network_from_onnx(name, batch_size):
    assert name in WORKLOADS_DIC
    # if batch_size > 1, we need to think about how to take care of bert and nasrnn
    assert batch_size == 1

    # Load
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = f"{this_code_path}/models/{name}.onnx"
    logging.info(f"loading ONNX model from {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)

    # Describe
    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    for input_info in onnx_model.graph.input:
        if input_info.name not in initializer_names:
            _, shape, dtype, _ = relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise ValueError(
                    f"Unknown dtype on input '{input_info.name}' is not supported."
                )
            input_shapes.update({input_info.name: shape})
            input_dtypes.update({input_info.name: dtype})
    logging.info(f"Loaded model {onnx_model_path} has shapes {input_shapes} and types {input_dtypes}")

    # Set the input shape dict
    shape_dict = WORKLOADS_DIC[name][batch_size]
    logging.info(f"Importing model {onnx_model_path} with concrete shapes {shape_dict}")

    # Import
    # We should copy shape_dict because shape_dict will be consumed in from_onnx
    shape_dict_tmp = copy.deepcopy(shape_dict)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict_tmp, freeze_params=True)
    mod = relay.transform.InferType()(mod)  # useful to see types of intermediate ops to assess BYOC applicability
    logging.info(f"Imported model {onnx_model_path} is:\n{mod}")

    return mod, params, shape_dict, None  # we don't need output shape
