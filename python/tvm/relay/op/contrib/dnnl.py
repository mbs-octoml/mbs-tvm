# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import Call, Constant, Tuple, GlobalVar, Var, TupleGetItem

from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("DNNL")


def check_dynamism(args, op_name):
    """
    Check for dynamism inside any of the args in the op.

    Parameters
    ----------
    args : tvm.ir.container.Array
        Arguments of the op. Each of the argument shape is checked for presence of dynamic
        components.
    op_name: str
        Name of the op for debugging purposes only.
    Returns
    ----------
    ret : bool
        True if dynamism is present, False otherwise
    """
    for arg in args:
        if isinstance(arg, (Call, Var, Constant, TupleGetItem)):
            for dim_shape in arg.checked_type.shape[1:]:
                if isinstance(dim_shape, tvm.tir.expr.Any):
                    return True
        elif isinstance(arg, Tuple):
            return check_dynamism(arg.fields, op_name)
        else:
            logger.info(
                "Arg not supported in DNNL for %s with type %s",
                op_name,
                type(arg),
            )
            return True
    return False

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

def _register_external_op_helper_with_checker(op_name, checker):
    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        attrs, args = expr.attrs, expr.args
        # ops with dynamic shapes are offloaded to VM
        if check_dynamism(args, op_name):
            return False
        if any([x.checked_type.dtype != "float32" for x in args]):
            logger.info("Only float32 inputs are supported for DNNL.")
            return False

        return checker(attrs, args, op_name)

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.dense")


def dim_check_fn(attrs, args, op_name):
    shapes = [[int(x) if not isinstance(x, tvm.tir.expr.Any) else -1 for x in arg.checked_type.shape]   for arg in args]
    if op_name == 'add':
        for shape in shapes:
            if len(shape) < 2:
                return False

            # @sunggg: Temp solution. Selectively disable adds in NasNetA
            if shape == [1, 64, 56, 56] or shape == [1,128,28,28] or shape == [1, 256, 14, 14]:
                return False


    elif op_name == 'nn.relu':
        for shape in shapes:
            if len(shape)!=4:
                return False
    else:
        raise Exception(f"Unsupported op for dim_check_fn {op_name}")

    return True


_register_external_op_helper_with_checker("add", dim_check_fn)
_register_external_op_helper_with_checker("nn.relu", dim_check_fn)

#_register_external_op_helper("tanh")
#_register_external_op_helper("sigmoid")
#_register_external_op_helper("subtract")
#_register_external_op_helper("multiply")

def make_conv_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.conv2d.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.conv2d`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


def make_dnnl_pattern(op, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = "dnnl." + op
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if op == "conv2d":
        dnnl_pattern = (pat_name, make_conv_pattern(with_bias, with_eltwise))
    elif op == "dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning("Currently, only conv2d and dense op are supported, but got %s.", op)
        dnnl_pattern = ()
    return dnnl_pattern


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    elt_list = ["nn.relu", "tanh", "sigmoid", None]
    dnnl_patterns = []
    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                return dnnl_patterns
            dnnl_patterns.append(make_dnnl_pattern("conv2d", with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("dense", with_bias, elt))
    return dnnl_patterns


def partition_for_dnnl(mod, params=None):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.CanonicalizeOps(),
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
            transform.SimplifyExpr(),
            transform.FoldConstant(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
