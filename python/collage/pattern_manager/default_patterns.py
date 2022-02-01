import tvm
from enum import Enum, auto
from tvm.relay.dataflow_pattern import (
    is_op,
    wildcard,
    is_tuple_get_item,
    is_tuple,
    is_constant
)

from .pattern_language import Pattern
from collage.utils import get_diamond

# Warning(@Soo): note that we ignore tuplegetitem nodes in TVM Relay,
# because they are only used to extract result of Relay's batch_norm operator
# maps op type to pattern representing it
str_to_pattern = {
    # RESNE(X)T
    "ADD": Pattern(is_op('add')(wildcard(), wildcard())),
    "CONV2D": Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
    "CONV2D_WINOGRAD_WO_WT": Pattern(
        is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard())),
    "RELU": Pattern(is_op("nn.relu")(wildcard())),
    "CONV2D_RELU": Pattern(is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))),
    "CONV2D_WINOGRAD_WO_WT_RELU": Pattern(
        is_op("nn.relu")(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard()))),
    "CONV2D_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
    # "ADD_RELU" : Pattern(is_op("nn.relu")(is_op("add")(wildcard(), wildcard()))),

    # BERT
    "DENSE": Pattern(is_op("nn.dense")(wildcard(), wildcard())),
    "RESHAPE": Pattern(is_op("reshape")(wildcard())),
    "TRANSPOSE": Pattern(is_op("transpose")(wildcard())),
    "BATCH_MATMUL": Pattern(is_op("nn.batch_matmul")(wildcard(), wildcard())),
    "RESHAPE_TRANSPOSE": Pattern(is_op("transpose")(is_op("reshape")(wildcard()))),
    "TRANSPOSE_RESHAPE": Pattern(is_op("reshape")(is_op("transpose")(wildcard()))),
    "DENSE_RELU": Pattern(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))),

    # NASRNN
    "TANH": Pattern(is_op("tanh")(wildcard())),
    "SIGMOID": Pattern(is_op("sigmoid")(wildcard())),
    "MULTIPLY": Pattern(is_op("multiply")(wildcard(), wildcard())),
    "TUPLE_GET_ITEM_0": Pattern(is_tuple_get_item(wildcard(), 0)),
    "TUPLE_GET_ITEM_1": Pattern(is_tuple_get_item(wildcard(), 1)),
    "TUPLE_TWO_IDX": Pattern(is_tuple([wildcard(), wildcard()])),
    "DENSE_RELU_ADD_SIGMOID": Pattern(
        is_op("sigmoid")(is_op("add")(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard())), is_constant()))),
    "DENSE_RELU_ADD_TANH": Pattern(
        is_op("tanh")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
    "DENSE_RELU_ADD_RELU": Pattern(
        is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
    "MULTIPLY_TANH": Pattern(is_op("tanh")(is_op("multiply")(wildcard(), wildcard()))),
    "RELU_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(wildcard())))),
    "ADD_SIGMOID": Pattern(is_op("sigmoid")(is_op("add")(wildcard(), wildcard()))),
    "ADD_TANH": Pattern(is_op("tanh")(is_op("add")(wildcard(), wildcard()))),

    # NASNET-A
    "CONCAT": Pattern(is_op("concatenate")(wildcard())),
    "BIAS_ADD": Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
    "AVG_POOL2D": Pattern(is_op("nn.avg_pool2d")(wildcard())),
    "MAX_POOL2D": Pattern(is_op("nn.max_pool2d")(wildcard())),
    "TUPLE_FIVE_IDX": Pattern(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()])),
    "CONV2D_BIAS_RELU": Pattern(
        is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), is_constant()))),
    "CONV2D_ADD": Pattern(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard())),
    "AVG_POOL2D_ADD": Pattern(is_op("add")(is_op("nn.avg_pool2d")(wildcard()), wildcard())),
    "TUPLE_FIVE_IDX_CONCAT": Pattern(
        is_op("concatenate")(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()]))),

    # ResNet-3D
    "CONV3D": Pattern(is_op("nn.conv3d")(wildcard(), wildcard())),
    "CONV3D_RELU": Pattern(is_op("nn.relu")(is_op("nn.conv3d")(wildcard(), wildcard()))),
    "CONV3D_ADD": Pattern(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard())),
    "CONV3D_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard()))),

    # Others
    "DIAMOND": get_diamond(),
    "BATCHNORM": Pattern(
        is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
    "SOFTMAX": Pattern(is_op("nn.softmax")(wildcard())),
    "BATCH_FLATTEN": Pattern(is_op("nn.batch_flatten")(wildcard())),
    "GLOBAL_AVG_POOL2D": Pattern(is_op("nn.global_avg_pool2d")(wildcard())),
    "CONV3D_BIAS_RELU": Pattern(
        is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv3d")(wildcard(), wildcard()), is_constant()))),

    # Other Fused Ops
    "CONV2D_BN": Pattern(is_tuple_get_item(
        is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(),
                               wildcard()), 0)),
    "BN_RELU": Pattern(is_op("nn.relu")(
        is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
    "CONV2D_BN_RELU": Pattern(is_op("nn.relu")(is_tuple_get_item(
        is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(),
                               wildcard()), 0))),
    "SUBTRACT": Pattern(is_op("subtract")(wildcard(), wildcard())),
    "EXPAND_DIMS": Pattern(is_op("expand_dims")(wildcard()))
}

# maps relay operator type to names of input vars.
relayop_to_varnames = {
    "nn.pad": ["data", "bias"],
    # RESNE(X)T
    "add": ["data", "data"],
    "nn.conv2d": ["data", "weight"],
    "nn.contrib_conv2d_winograd_without_weight_transform": ["data", "weight"],
    "nn.relu": ["data"],

    # BERT
    "nn.dense": ["data", "weight"],
    "reshape": ["data"],
    "transpose": ["data"],
    "nn.batch_matmul": ["data", "data"],
    # "nn.batch_matmul" : ["x", "y"],

    # NASRNN
    "tanh": ["data"],
    "multiply": ["data", "data"],
    # "multiply": ["lhs", "rhs"],
    "sigmoid": ["data"],
    # FIXME(@Soo): How should we deal with TUPLE and TUPLE_GET_ITEM?

    # NASNET-A
    "concatenate": ["data"],
    "nn.bias_add": ["data", "bias"],
    "nn.avg_pool2d": ["data"],
    "nn.max_pool2d": ["data"],
    "tuple": ["data", "data", "data", "data", "data"],

    # RESNET_3D
    "nn.conv3d": ["data", "weight"],

    # Others
    "nn.batch_norm": ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
    "nn.softmax": ["data"],
    "nn.batch_flatten": ["data"],
    "nn.global_avg_pool2d": ["data"],

    # DCGAN
    "image.resize": ["data"],

    # BERT_FULL
    "divide": ["data", "data"],
    "subtract": ["data", "data"],
    "sqrt": ["data"],
    "variance": ["data", "data"],
    "mean": ["data"],
    "expand_dims" : ["data"],
}


def convert_str_to_pattern(lst):
    from collage.pattern_manager.default_patterns import str_to_pattern
    return [
        tuple([str_to_pattern[name], constraint_func])
        for name, constraint_func in lst
    ]


# cuDNN
# Set of pattern and its constraints
cudnn_default_patterns_str = [
    ["CONV2D", None],
    ["CONV3D", None],
    ["SOFTMAX", None],
    ["MAX_POOL2D", None],
    ["AVG_POOL2D", None],
    # ["CONV2D_ADD_RELU", None],
    # ["RELU", None],
    ["CONV3D_BIAS_RELU", None],
    ["BATCHNORM", None],
]
cudnn_default_patterns = convert_str_to_pattern(cudnn_default_patterns_str)

# cuBLAS
cublas_default_patterns_str = [
    ["DENSE", None],
    ["BATCH_MATMUL", None],
]
cublas_default_patterns = convert_str_to_pattern(cublas_default_patterns_str)


# DNNL
def dnnl_check_add(config):
    for idx_shape, shape in enumerate(config._data_shape):
        if len(shape) < 2:
            return False

        # Check if all inputs have same dimensionality
        if idx_shape > 0 and len(shape) != prev_shape:
            return False
        prev_shape = len(shape)

        if shape == [1, 64, 56, 56] or shape == [1, 128, 28, 28] or shape == [1, 256, 14, 14]:
            return False
    return True


def dnnl_check_relu(config):
    for shape in config._data_shape:
        if len(shape) != 4:
            return False
    return True


dnnl_default_patterns_str = [
    ["CONV2D", None],
    ["CONV3D", None],
    ["BATCHNORM", None],
    ["DENSE", None],
    # ["ADD", dnnl_check_add]
    ["RELU", dnnl_check_relu],
]
dnnl_default_patterns = convert_str_to_pattern(dnnl_default_patterns_str)


# MKL
def mkl_check_dense(config):
    dim1 = len(config._data_shape[0])
    dim2 = len(config._data_shape[1])
    return dim1 == 2 and dim2 == 2


mkl_default_patterns_str = [
    ["DENSE", mkl_check_dense],
    ["BATCH_MATMUL", None],
]
mkl_default_patterns = convert_str_to_pattern(mkl_default_patterns_str)

# TODO(mbs): Can't directly import BYOC patterns since re-implementation of pattern machinery does not support Alt.
# cutlass_default_patterns = [(Pattern(pat), pred) for (label, pat, pred) in tvm.relay.op.contrib.cutlass_patterns()]

def cutlass_check_dense(config):
    return tvm.relay.op.contrib.cutlass.check_gemm(config.expr)

def cutlass_check_conv2d(config):
    return tvm.relay.op.contrib.cutlass.check_conv2d(config.expr)

cutlass_default_patterns_str = [
    ["DENSE", cutlass_check_dense],
    ["CONV2D", cutlass_check_conv2d],
    ["CONV2D_RELU", cutlass_check_conv2d],
    ["CONV2D_ADD_RELU", cutlass_check_conv2d],
]
cutlass_default_patterns = convert_str_to_pattern(cutlass_default_patterns_str)
