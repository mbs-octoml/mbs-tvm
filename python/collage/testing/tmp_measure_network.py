from collage.optimizer.custom_fusion_pass import measure_end_to_end_user_defined
from workloads.torch_workloads import get_network_from_torch

import sys

assert(len(sys.argv) == 13)

net_name = sys.argv[1]
target_str = sys.argv[2]
batch_size = int(sys.argv[3])
autotvm_tuning_log = sys.argv[4]
backends = sys.argv[5].split(",")
op_cost_log_path = sys.argv[6]
op_level_placement_log = sys.argv[7]
graph_level_placement_log = sys.argv[8]
graph_level_tmp_file = sys.argv[9]
evolutionary_search_pop_size = int(sys.argv[10])
evolutionary_search_max_iter = int(sys.argv[11])
evolutionary_search_budget= float(sys.argv[12])


mod, params, shape_dict, _ = get_network_from_torch(net_name, batch_size)
perf, std = measure_end_to_end_user_defined(
                    mod["main"], 
                    params, 
                    shape_dict, 
                    target_str, 
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
            )

print("##result: ", perf, std, file=sys.stderr)
