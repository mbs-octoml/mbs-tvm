from workloads.torch_workloads import get_network_from_torch
import numpy as np
import collage
import tvm
import logging
from tvm.contrib import graph_executor as runtime
from collage.backend.default_backends import (
                    cg_AutoTVM,
                    cg_VanillaTVM,
                )
from collage.pattern_manager.base_pattern_rule import (
                    BasePatternRule, 
                    BasePatternGenerator,
                )
from collage.pattern_manager.default_pattern_rules import (
                    tvm_pattern_generator,
                    DefaultPatternGenerator,
                )
from tvm.relay.dataflow_pattern import (
                        is_op, 
                        wildcard, 
                        is_tuple_get_item, 
                        is_tuple, 
                        is_constant
                    )
from collage.pattern_manager.pattern_language import Pattern
from collage.utils import (
                        is_var_node, 
                        is_constant_node, 
                        is_tuple_node, 
                        is_tuplegetitem_node,
                        get_op_pattern,
                        is_call_node,
                        get_args,
                        is_var,
                    )

# [NOTE]
# * Available networks: bert_full, dcgan, nasneta, resnet50_3d, resnext50_32x4d, yolov3, mobilenet_v2
# * Collage supports following backends by default:
#      NVIDIDIA GPUs - TVM, TensorRT, cuBLAS, cuDNN
#      Intel CPUs    - TVM, MKL, DNNL
# * For the best performance, TVM operators should be tuned with AutoTVM beforehand.
# * Collage offers two optimizers: "op-level", "two-level"

# Define Collage workload
workload = {
    "optimizer": "op-level", 
    "backends": ["autotvm", "cudnn", "cublas", "tensorrt"], 
    "network_name": "dcgan", 
    "target": "cuda",
    "batch_size": 1,
}

# Default logging level. Skip messages during optimization
logging.basicConfig(level=logging.ERROR)

# Enable logging to monitor optimization progress e.g., operator matching, profiling...
#logging.basicConfig(level=logging.INFO)

def measure_perf(lib, workload):
    # Create workload
    dev = tvm.device(workload["target"], 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in workload["shape_dict"].items():
        input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    # Measure performance
    ftimer = module.module.time_evaluator("run", dev, number=10, repeat=20)
    perfs = np.array(ftimer().results) * 1000
    return np.mean(perfs), np.std(perfs)


def setup_workload(workload):
    network_name, batch_size, target = \
          workload["network_name"], workload["batch_size"], workload["target"]

    mod, params, shape_dict, _ = get_network_from_torch(network_name, batch_size)
    # Since Collage utilizes tvm as its codegen, we need to pass the following info for tvm codegen.
    workload["mod"] = mod
    workload["params"] = params
    workload["shape_dict"] = shape_dict


if __name__ == "__main__":
    setup_workload(workload)

    # Create the collage module
    collage_mod = collage.Module()
    print(f"Default backends: {collage_mod.get_registered_backends()}\n")
     # Override the default tuning log
    collage_mod.update_backend_tuning_log("autotvm", "autotvm_tuning_log_rtx2070.json")

    # 1. Register new backend with the direct sepecification
    # With this approach, users pass the list of pattern and its constraints
    # As a codegen, this example uses AutoTVM that collage provide by default
    def check_dimension(config):
        dim1 = len(config._data_shape[0])
        dim2 = len(config._data_shape[1])
        return dim1 == 2 and dim2 == 2

    patterns = [
        tuple([Pattern(is_op("nn.conv2d")(wildcard(), wildcard())), check_dimension]),
        tuple([Pattern(is_op("nn.dense")(wildcard(), wildcard())), None])
    ]

    collage_mod.register_new_backend(
                                        name="SimpleBackend", 
                                        kind=collage.BackendKind.OP_LEVEL,
                                        codegen=cg_AutoTVM, 
                                        patterns=patterns
                                    )
    new_patterns = [
        tuple([Pattern(is_op("nn.relu")(wildcard())), None])
    ]
    collage_mod.add_backend_patterns("SimpleBackend", new_patterns)
    # Since 'SimpleBackend' uses AutoTVM backend, feed tuning log
    collage_mod.update_backend_tuning_log("SimpleBackend", "autotvm_tuning_log_rtx2070.json")

    print(f"Backend Patterns in SimpleBackend: {collage_mod.get_backend_patterns('SimpleBackend')}")


    # 2. Users can bring their custom codegen for Collage
    # This example registers TVM codegen (w/o tuning) with a default pattern generator that collage provides
    def cg_VanillaTVM(net, target, params, **kwargs):
        with tvm.transform.PassContext(opt_level=3):
            lib = tvm.relay.build_module.build(net, target=target, params=params)
        return lib

    collage_mod.register_new_backend(
                                        name="VanillaTVM", 
                                        kind=collage.BackendKind.OP_LEVEL,
                                        codegen=cg_VanillaTVM, 
                                        pattern_generator=tvm_pattern_generator
                                    )

    print(f"Registered backends: {collage_mod.get_registered_backends()}")

    # Run backend placement optimization with two custom backends
    workload["backends"] = ["VanillaTVM", "SimpleBackend"]
    lib = collage_mod.optimize_backend_placement(**workload)
    collage_mean_perf, collage_std_perf = measure_perf(lib, workload)
    print(f"# Network: {workload['network_name']}, Collage optimizer: {workload['optimizer']}")
    print(f"    - Provided backends: {workload['backends']}")
    print(f"    - Run with Collage  (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")
    

    # 3. Register new backend with a pattern rule
    # Operator patterns generated by this rule should be supported by the backing codegen
    # This demo uses TVM codegen with AutoTVM
    # This example will consider conv2d as a seed and fuse their following elemwise operators 
    # e.g., O: conv2d+relu, conv2d+add+relu, ...   
    #       X: add+relu, add+sigmoid, ...  - there is no seed node
    class CustomFusionRule(BasePatternRule):
        __instance = None
        # Define seed operator
        SEED_OPS = set(["nn.conv2d"])
        # For simplicity, we only consider following operators as elemwise-ops.
        ELEM_OPS = set(["nn.relu", "tanh", "signmoid", "add"])
        # Define max number of operators that can be fused
        MAX_NUM_OPS = 10

        @staticmethod
        def destory():
            CustomFusionRule.__instance = None
        
        def __init__(self):
            """ Virtually private constructor. """
            if CustomFusionRule.__instance != None:
                raise Exception("This class should be a singleton!")
            CustomFusionRule.__instance = self

        # NOTE: Calling convention of op_rule and fusion_rule will be determined by pattern generator that will be used.
        # This example will use default pattern generator

        # Define valid operators
        # Users can add any constraints to filter out unsupported operators efficiently
        @staticmethod
        def op_rule(expr):
            # Only support either seed ops or elemwise ops
            if is_call_node(expr):
                return (
                            expr.op.name in CustomFusionRule.SEED_OPS or 
                            expr.op.name in CustomFusionRule.ELEM_OPS
                        )
            return False

        # Default pattern generator that tries to fuse all operators between src and sink nodes if possible
        # Here, sink node will be post-dominant node of src node
        @staticmethod
        def fusion_rule(src, sink, cur_type, num_ops):
            SEED_OPS, ELEM_OPS, MAX_NUM_OPS = CustomFusionRule.SEED_OPS, CustomFusionRule.ELEM_OPS, CustomFusionRule.MAX_NUM_OPS
            # If number of operators are greater than MAX_NUM_OPS, stop exapnding fusion group
            if num_ops > MAX_NUM_OPS:
                return list()

            # Check operators between src and sink node in dfs order if they satisfies fcheck
            # If they do, pass the operators on paths between src and sink
            # Pattern generator will generate the pattern accordingly and register them to pattern registry 
            def _check_path(src, sink, fcheck):
                def helper(src, node, fcheck, path = [], paths = []):
                    path.append(node)

                    if src == node:
                        assert(len(path))    
                        paths.append(path.copy())
                    elif is_var_node(node) or is_constant_node(node):
                        pass
                    elif fcheck(node):
                        children = []
                        if is_tuple_node(node):
                            children = node.fields
                        elif is_tuplegetitem_node(node):
                            children = [ node.tuple ]
                        elif is_call_node(node):
                            children = node.args
                        else:
                            raise Exception(f"Unsupported type ({type(node)})")
                    
                        for child in children:
                            helper(src, child, fcheck, path, paths)
                
                    out = path.pop()
                    assert(node == out)
                
                path, paths = [], []
                helper(src, sink, fcheck, path, paths)
                
                return paths
            
            if src.op.name in SEED_OPS:
                def fcheck(node):
                    return is_call_node(node) and node.op.name in ELEM_OPS
                
                return _check_path(src, sink, fcheck)
            
            return list()

    # Instantiate pattern rule object and create the default pattern generator with it 
    custom_pattern_generator = DefaultPatternGenerator(CustomFusionRule())
    # Register new backend
    collage_mod.register_new_backend(
                                        name="CustomFusion", 
                                        kind=collage.BackendKind.OP_LEVEL,
                                        codegen=cg_AutoTVM, 
                                        pattern_generator=custom_pattern_generator
                                    )
    # Since we are using AutoTVM codegen, feed tuning log
    collage_mod.update_backend_tuning_log("CustomFusion", "autotvm_tuning_log_rtx2070.json")
    print(f"Registered backends: {collage_mod.get_registered_backends()}")
    
    # Run Collage with three custom backends
    workload["backends"] = ["VanillaTVM", "SimpleBackend", "CustomFusion"]
    lib = collage_mod.optimize_backend_placement(**workload)
    collage_mean_perf, collage_std_perf = measure_perf(lib, workload)

    print(f"# Network: {workload['network_name']}, Collage optimizer: {workload['optimizer']}")
    print(f"    - Provided backends: {workload['backends']}")
    print(f"    - Run with Collage  (mean, std) = ({collage_mean_perf:.4f}+-{collage_std_perf:.4f})")

    # Visualize backend placement optimized by op-level optimizer
    # If two-level optimization is enabled, users can also pass 'workload["input_placement_log_file"] = collage_mod.graph_level_placement_log'
    workload["input_placement_log_file"] = collage_mod.op_level_placement_log
    workload["placement_vis_file"] = "demo_customization"
    collage_mod.visualize_backend_placement(**workload)