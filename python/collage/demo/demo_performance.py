import numpy as np
import collage
from collage.workloads.torch_workloads import get_network_from_torch
import tvm
import logging
from tvm.contrib import graph_executor as runtime

# [NOTE]
# * Available networks: bert_full, dcgan, nasneta, resnet50_3d, resnext50_32x4d, yolov3, mobilenet_v2
# * Collage supports following backends by default:
#      NVIDIDIA GPUs - TVM, TensorRT, cuBLAS, cuDNN
#      Intel CPUs    - TVM, MKL, DNNL
# * For the best performance, TVM operators should be tuned with AutoTVM beforehand.
# * Collage offers two optimizers: "op-level", "two-level"
#   Since two-level optimization takes long time (~30min), op-level optimizer is configured by default in this demo.

# Define Collage workload
workload = {
    "optimizer": "op-level", 
    "backends": ["autotvm", "cudnn", "cublas", "tensorrt"],
    "network_name": "bert",
    "target": tvm.target.Target("nvidia/geforce-rtx-3070"),
    "batch_size": 1,
    "tuning_log": "autotvm_tuning_log_bert_full_rtx3070.json",
}

# Default logging level. Skip messages during optimization
# logging.basicConfig(level=logging.ERROR)

# Enable logging to monitor optimization progress e.g., operator matching, profiling...
logging.basicConfig(level=logging.INFO)

def measure_perf(lib, workload):
    # Create workload
    dev = tvm.device(workload["target"].kind.device_type, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in workload["shape_dict"].items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)

def run_with_tensorrt(workload):
    from collage.backend.default_backends import cg_TensorRT
    lib = cg_TensorRT(workload["mod"], workload["target"], workload["params"])
    return measure_perf(lib, workload)


def setup_workload(workload):
    network_name, batch_size, target = \
          workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen.
    workload["mod"] = mod
    workload["params"] = params
    workload["shape_dict"] = shape_dict

    print(mod)

if __name__ == "__main__":
    setup_workload(workload)
    # Measure TensorRT performance as baseline
    trt_mean_perf, trt_std_perf = run_with_tensorrt(workload)

    # Operator cost will be logged at "operator_cost.log" by default.
    # If you want to start from scratch, delete previous log file for operator cost.
    # Since it is unreadable, users can dump human-readable form by passing 'dump_readable_cost_log = True'
    collage_mod = collage.Module(op_cost_log_path = "operator_cost.log", dump_readable_cost_log = False)
    print(f"Default backends: {collage_mod.get_registered_backends()}\n")

    # Override the default tuning log
    # If you don't have tuning log, generate one by running 'autotune_tvm_ops.py'
    collage_mod.update_backend_tuning_log("autotvm", workload["tuning_log"])

    # Invoke collage optimizer
    lib = collage_mod.optimize_backend_placement(**workload)
    collage_mean_perf, collage_std_perf = measure_perf(lib, workload)

    print(f"[ End-to-End Performance Evaluation ]")
    print(f"# Performance of Collage is compared against TensorRT")
    print(f"  speedup = (performance of TensorRT)/(performance of Collage)")
    print(f"\n")
    print(f"# Network: {workload['network_name']}, Collage optimizer: {workload['optimizer']}")
    print(f"  * End-to-end performance")
    print(f"    - Run with TensorRT (mean, std) = ({trt_mean_perf:.4f}+-{trt_std_perf:.4f})")
    print(f"    - Run with Collage  (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f}), Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")

    # Visualize backend placement optimized by op-level optimizer
    # If two-level optimization is enabled, users can also pass 'workload["input_placement_log_file"] = collage_mod.graph_level_placement_log'
    workload["input_placement_log_file"] = collage_mod.op_level_placement_log
    workload["placement_vis_file"] = "demo_performance"
    collage_mod.visualize_backend_placement(**workload)

    if False:
        print(f"\n  * Ablation study")
        workload["backends"] = ["autotvm"]
        lib = collage_mod.optimize_backend_placement(**workload)
        collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
        print(f"    - Run with Collage  w/ 1 backend  (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f}), Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")

        workload["backends"] = ["autotvm", "cudnn"]
        lib = collage_mod.optimize_backend_placement(**workload)
        collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
        print(f"    - Run with Collage  w/ 2 backends (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f}), Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")

        workload["backends"] = ["autotvm", "cudnn", "cublas"]
        lib = collage_mod.optimize_backend_placement(**workload)
        collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
        print(f"    - Run with Collage  w/ 3 backends (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f}), Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")

        workload["backends"] = ["autotvm", "cudnn", "cublas", "tensorrt"]
        lib = collage_mod.optimize_backend_placement(**workload)
        collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
        print(f"    - Run with Collage  w/ 4 backends (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f}), Speedup: {trt_mean_perf/collage_mean_perf:.4f}x")
