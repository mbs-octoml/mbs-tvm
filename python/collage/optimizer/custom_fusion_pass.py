from enum import IntEnum
from collage.utils import is_function_node
from collage.measurer.base import (
            NUM_MEASUREMENTS_PER_REPEAT,
            NUM_REPEATS,
            measure,
        )
import tvm
from tvm import relay, autotvm

from collage.interface import CollageContext
import collage
import tvm.contrib.graph_executor as runtime
import numpy as np

CONFIG_VAR_USER_DEFINED_FUSION_PASS = "relay.FuseOps.UserDefinedFusion"

class CustomFusionPass(IntEnum):
    # This is for measurement
    USER_DEFINED_FUSION = 0
    DP = 1
    EXHAUSTIVE_SEARCH = 2
    TWO_LEVEL_OPT = 3
    OP_MEASUREMENT = 4
    SINGLE_BACKEND_BASELINE = 5
    VISUALIZE_BACKEND_PLACEMENT = 6

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_



def get_opt_info_tag(net_name, hw_name, batch_size):
    return f"{net_name}_{hw_name}_bs{batch_size}"

#def get_best_match_file_name(net_name, hw_name, batch_size):
#    opt_info_tag = get_opt_info_tag(net_name, hw_name, batch_size)
#    return f"{BEST_MATCH_LOG}_{opt_info_tag}"

#def get_user_defined_match_path(net_name, hw_name, batch_size):
#    opt_info_tag = get_opt_info_tag(net_name, hw_name, batch_size)
#    return f"{LOG_PATH}/user_defined_match_{opt_info_tag}.log"


def measure_end_to_end_user_defined(
            net, 
            params, 
            shape_dict, 
            build_target, 
            net_name, 
            batch_size, 
            autotvm_tuning_log, 
            backends,
            op_cost_log_path,
            op_level_placement_log,
            graph_level_placement_log, 
            graph_level_tmp_file,
            evolutionary_search_pop_size,
            evolutionary_search_max_iter,
            evolutionary_search_budget
        ):
    assert is_function_node(net)

    net = net.with_attr("CustomFusionPass", CustomFusionPass.USER_DEFINED_FUSION)
    net = net.with_attr("Network", net_name)
    net = net.with_attr("BuildTarget", build_target)
    net = net.with_attr("BatchSize", batch_size)

    with CollageContext(
                collage.Module(op_cost_log_path), 
                backends,
                op_level_placement_log,
                graph_level_placement_log, 
                graph_level_tmp_file,
                evolutionary_search_pop_size,
                evolutionary_search_max_iter,
                evolutionary_search_budget
            ):
        with autotvm.apply_history_best(autotvm_tuning_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(net, build_target, params=params)
            # Create workload
            dev = tvm.device(build_target, 0)
            module = runtime.GraphModule(lib["default"](dev))

            # Setup execution
            for input_name, input_shape in shape_dict.items():
                input_data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
                module.set_input(input_name, input_data)

            ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    return measure(ftimer, build_target)


