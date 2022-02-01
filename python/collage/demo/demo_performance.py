import numpy as np
import collage
from collage.workloads.onnx_workloads import get_network_from_onnx
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
    "backends": ["autotvm", "cudnn", "tensorrt", "cutlass", "cublas"],
    "network_name": "gpt2",
    "target": tvm.target.Target("nvidia/geforce-rtx-3070"),
    "batch_size": 1,
    "tuning_log": "autotvm_tuning_log_mnist_rtx3070.json",  ###### CHANGE ######
}

logging.basicConfig(level=logging.INFO)


def measure_perf(lib, workload):
    # Create workload
    dev = tvm.device(workload["target"].kind.device_type, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in workload["shape_dict"].items():
        input_data = np.random.uniform(-1, 1,
                                       size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)


def run_with_tvm(workload):
    from collage.backend.default_backends import cg_AutoTVM
    lib = cg_AutoTVM(workload["mod"], workload["target"], workload["params"], tuning_log=workload["tuning_log"])
    return measure_perf(lib, workload)


def run_with_tensorrt(workload):
    from collage.backend.default_backends import cg_TensorRT
    lib = cg_TensorRT(workload["mod"], workload["target"], workload["params"])
    return measure_perf(lib, workload)


def setup_workload(workload):
    network_name, batch_size, target = \
        workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_onnx(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen.
    workload["mod"] = mod
    workload["params"] = params
    workload["shape_dict"] = shape_dict


if __name__ == "__main__":
    setup_workload(workload)
    # Measure baseline performances
    logging.info("Timing with autotvm")
    tvm_mean_perf, tvm_std_perf = run_with_tvm(workload)
    logging.info("Timing with BYOC(tensorrt)")
    trt_mean_perf, trt_std_perf = run_with_tensorrt(workload)

    # Operator cost will be logged at "operator_cost.log" by default.
    # If you want to start from scratch, delete previous log file for operator cost.
    # Since it is unreadable, users can dump human-readable form by passing 'dump_readable_cost_log = True'
    collage_mod = collage.Module(op_cost_log_path="operator_cost.log", dump_readable_cost_log=False)

    # Override the default tuning log
    # If you don't have tuning log, generate one by running 'autotune_tvm_ops.py'
    collage_mod.update_backend_tuning_log("autotvm", workload["tuning_log"])

    # Invoke collage optimizer
    logging.info(f"Timing with Collage({workload['backends']})")
    lib = collage_mod.optimize_backend_placement(**workload)
    collage_mean_perf, collage_std_perf = measure_perf(lib, workload)

    print(f"Network:                         {workload['network_name']}")
    print(f"Collage optimizer:               {workload['optimizer']}")
    print(f"AutoTVM  (mean, std):            ({tvm_mean_perf:.4f}+-{tvm_std_perf:.4f})")
    print(f"TensorRT (mean, std):            ({trt_mean_perf:.4f}+-{trt_std_perf:.4f})")
    print(f"Collage  (mean, std):            ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")
    print(f"Collage speedup w.r.t. AutoTVM:  {tvm_mean_perf / collage_mean_perf:.4f}x")
    print(f"Collage speedup w.r.t. TensorRT: {trt_mean_perf / collage_mean_perf:.4f}x")

    if False:
        # Visualize backend placement optimized by op-level optimizer
        # If two-level optimization is enabled, users can also pass 'workload["input_placement_log_file"] = collage_mod.graph_level_placement_log'
        workload["input_placement_log_file"] = collage_mod.op_level_placement_log
        workload["placement_vis_file"] = "demo_performance"
        collage_mod.visualize_backend_placement(**workload)
