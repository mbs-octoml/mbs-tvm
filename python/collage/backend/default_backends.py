import os
import tvm
from tvm import relay, autotvm
from tvm.relay import transform
from tvm.contrib.cutlass import (
    tune_cutlass_kernels,
    build_cutlass_kernels,
)

EXTERNAL_COMPILERS = ['tensorrt', 'dnnl']
XEON_BUILD_TARGET = 'llvm -mcpu=skylake-avx512'
NVIDIA_GPUS = ['rtx2070', 'rtx3070', 'jetson', 'v100']
INTEL_CPUS = ['xeon']


# codegens
def cg_AutoTVM(net, target, params, **kwargs):
    assert("tuning_log" in kwargs)
    tuning_log = kwargs["tuning_log"]
    assert(os.path.exists(tuning_log))
    # AutoTVM codes
    # Compile kernels with history best records
    with autotvm.apply_history_best(tuning_log):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(net, target=target, params=params)
    return lib

def cg_VanillaTVM(net, target, params, **kwargs):
    # TVM without auto-tuning
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(net, target=target, params=params)
    return lib


def cg_TensorRT(mod, target, params, **kwargs):
    from tvm.relay.build_module import bind_params_by_name
    from tvm.relay.op.contrib.tensorrt import RemoveDropoutPass

    config = {
        "use_implicit_batch": False,
        "max_workspace_size": 1 << 30,
        "remove_no_mac_subgraphs": False,
    }
    
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            RemoveDropoutPass(),
            transform.RemoveUnusedFunctions(),
            transform.ConvertLayout(
                {
                    "nn.conv2d": ["NCHW", "default"],
                    "nn.conv3d": ["NCDHW", "default"],
                    "nn.conv2d_transpose": ["NCHW", "default"],
                }
            ),
            transform.FoldConstant(),
            transform.AnnotateTarget("tensorrt"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )
    
    # Annotate
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        mod = seq(mod)
    
    # We confirm that TVM can't pass conv2d to TensorRT if it's winograd without wt
    with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
        lib = relay.build(mod, target=target, params=params)

    #lib.export_library('compiled_tensorrt.so')
    #loaded_lib = tvm.runtime.load_module('compiled_tensorrt.so')
    return lib


def cg_DNNL(net, target, params, **kwargs):
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        raise Exception("skip because DNNL codegen is not available")
        return
    opt_pass = tvm.transform.Sequential(
            [
                tvm.relay.transform.InferType(),
                tvm.relay.transform.SimplifyInference(),
                tvm.relay.transform.FoldConstant(),
                tvm.relay.transform.FoldScaleAxis(),
                tvm.relay.transform.AnnotateTarget("dnnl"),
                tvm.relay.transform.MergeCompilerRegions(),
                tvm.relay.transform.PartitionGraph(),
                tvm.relay.transform.InferType(),
            ]
        )

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = opt_pass(mod)

    # It's ok not to do AlterOpLayout because DNNL ops are gonna be changed to GlobalVar,
    # which won't be touched by AlterOpLayout
    with tvm.transform.PassContext(opt_level=3):#, disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target=target, params=params)

    return lib


# [NOTE]
# Since current TVM has no easy way to force lowering for the operator-level libraries, we built a hacky way.
def cg_op_level_backends(net, target, params, **kwargs):
    assert("annotation" in kwargs)
    if kwargs["annotation"] == 'mkl':
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            lib = relay.build_module.build(net, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(net, target=target, params=params)
    return lib


def cg_cuDNN(net, target, params, **kwargs):
    return cg_op_level_backends(net, target, params, **kwargs)


def cg_cuBLAS(net, target, params, **kwargs):
    return cg_op_level_backends(net, target, params, **kwargs)


def cg_MKL(net, target, params, **kwargs):
    return cg_op_level_backends(net, target, params, **kwargs)


def cg_CUTLASS(mod, target, params, **kwargs):
    sm = 80   # rtx3070 is 8.6 but CUTLASS only supports 8.0
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = relay.op.contrib.partition_for_cutlass(mod, params)

        mod, num_cutlass_partition = tune_cutlass_kernels(mod, sm)

        lib = relay.build(mod, target=target, params=params)
        lib = build_cutlass_kernels(lib, sm)

    return lib
