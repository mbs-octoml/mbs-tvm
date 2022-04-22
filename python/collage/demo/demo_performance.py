import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime
import menangerie

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
    "model": menangerie.resnext50_32x4d(),
    "target": tvm.target.Target("cuda"),
    "batch_size": 1,
    "tuning_log": "/home/mbs/github/mbs-tvm/python/collage/demo/collage_autotvm_rtx3070.tuninglog",
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


def setup_workload(workload):
    workload["mod"] = workload["model"]["mod"]
    workload["params"] = workload["model"]["params"]
    workload["shape_dict"] = workload["model"]["input_shapes"]
    workload["network_name"] = workload["model"]["name"]


if __name__ == "__main__":
    setup_workload(workload)

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
    print(f"Collage  (mean, std):            ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")
