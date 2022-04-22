import tvm
from collage.utils import (is_var_node)
from collage.utils import printe
import os
import datetime

DATA_NAME_HINTS = ['data', 'input', 'x']


def log_matched_ops_by_method(log_path, matches):
    with open(log_path, 'w') as f:
        f.write("Matched operators (in post-dfs-order, from the root of comp graph to the last node)\n")
        for anno in matches:
            f.write(anno + "\n")


@tvm._ffi.register_func("collage.optimizer.visualize_expr")
def visualize_expr(expr, file_name):
    pass


def print_ir(mod, info, is_before):
    """Print the name of the pass, the IR, only before passes execute."""
    printe(f"Pass: {info.name}")
    if info.name in ["ToANormalForm", "InferType"]:
        return

    # if info.name == "AnnotateTargetFunc" or info.name == "MergeCompilerRegions" or info.name == "PartitionGraph":
    if is_before:
        printe("Running pass: {}", info.name)
    else:
        printe("Done pass: {}", info.name)


def is_data_node(expr):
    is_data_var = False
    for data_name_hint in DATA_NAME_HINTS:
        if data_name_hint in expr.name_hint:
            is_data_var = True
            break
    return is_data_var


def is_data_var_node(expr):
    if is_var_node(expr) and is_data_node(expr):
        return True

    return False


# def get_next_expr_after_match(relay_expr, prev_relay_expr, pattern, tmp_memo):
#     target_node = []
#
#     # Prevent diamond pattern from creating two same next elements and overwriting cost to 0
#     if relay_expr in tmp_memo:
#         return []
#     else:
#         tmp_memo[relay_expr] = True
#
#     if type(relay_expr) == tvm.relay.expr.Var:
#         if is_data_var_node(relay_expr):
#             return [(relay_expr, prev_relay_expr)]
#         assert False, "Var node other than data exists!"
#         # return [(None, prev_relay_expr)]
#     elif is_constant_node(relay_expr):
#         return [(None, prev_relay_expr)]
#
#     if isinstance(pattern, WildcardPattern):
#         return [(relay_expr, prev_relay_expr)]
#     # Warning(@Soo): This is hacky way
#     elif isinstance(pattern, ConstantPattern):
#         return [(None, prev_relay_expr)]
#
#     # If it is tuple, you should use tuple_value instead of args
#     # Caution: depth or depth-1?
#     if type(relay_expr) == tvm.relay.expr.TupleGetItem:
#         target_node += get_next_expr_after_match(relay_expr.tuple_value, relay_expr, pattern.tuple, tmp_memo)
#     elif is_tuple_node(relay_expr):
#         for f_idx, node in enumerate(relay_expr.fields):
#             target_node += get_next_expr_after_match(node, relay_expr, pattern.fields[f_idx], tmp_memo)
#     else:
#         # Note that batch_matmul also has args
#         # if type(relay_expr) == tvm.relay.nn.batch_matmul:
#         #     target_node += get_next_expr_after_match(relay_expr.x, relay_expr, depth - 1)
#         #     target_node += get_next_expr_after_match(relay_expr.y, relay_expr, depth - 1)
#         # else:
#         # print("Expr : ", relay_expr)
#         # print("Pattern : ", pattern)
#
#         # Warning(@Soo): pattern args are not necessarily in the same order with expr args
#         # e.g., is_op("add")(is_op("nn.conv2d"), wildcard) can still match add(data, conv)
#         # Thus, we need to figure out the order and call the function in right order
#         # Update (@Soo): With ordered_pattern_matcher, now the order is guaranteed.
#         for a_idx, node in enumerate(relay_expr.args):
#             target_node += get_next_expr_after_match(node, relay_expr, pattern.args[a_idx], tmp_memo)
# #             # FIX: Hacky way to avoid residual connection
# #             break
#
#     return target_node

def get_pattern_len(pattern):
    length = 0
    if type(pattern) == tvm.relay.dataflow_pattern.CallPattern:
        for child in pattern.args:
            length = max(length, get_pattern_len(child))
        length += 1
    elif type(pattern) == tvm.relay.dataflow_pattern.TupleGetItemPattern:
        length = get_pattern_len(pattern.tuple)
        length += 1
    elif type(pattern) == tvm.relay.dataflow_pattern.TuplePattern:
        for child in pattern.fields:
            length = max(length, get_pattern_len(child))
        length += 1

    return length


def print_matching_final(comp_graph, loc2match):
    idx = -1
    if hash(comp_graph._nodes[idx]) in loc2match:
        graph_str = loc2match[hash(comp_graph._nodes[idx])]["string"]
        reverse_graph_str = ""
        for node_str in graph_str[1:].split('-'):
            reverse_graph_str = node_str + "-" + reverse_graph_str
        print(f"Graph : {reverse_graph_str} (hash: {hash(comp_graph._nodes[idx])})")

        tot_cost = loc2match[hash(comp_graph._nodes[idx])]["cost"]
        print(f"Total Cost:{tot_cost}")

        print("Matched backend ops (op, cost)")
        for item in loc2match[hash(comp_graph._nodes[idx])]["match"][::-1]:
            op_name, op_cost, _ = item
            print(f"({op_name}, {op_cost:.2g})")
    else:
        raise Exception('Final matching does not exist.')


def print_matching_debug(comp_graph, loc2match):
    for idx in range(len(comp_graph._nodes)):
        if hash(comp_graph._nodes[idx]) in loc2match:
            graph_str = loc2match[hash(comp_graph._nodes[idx])]["string"]
            reverse_graph_str = ""
            for node_str in graph_str[1:].split('-'):
                reverse_graph_str = node_str + "-" + reverse_graph_str
            print(f"Graph : {reverse_graph_str} (hash: {hash(comp_graph._nodes[idx])})")

            tot_cost = loc2match[hash(comp_graph._nodes[idx])]["cost"]
            print(f"Total Cost:{tot_cost:.2g}")

            print("Matched backend ops (op, cost)")

            for item in loc2match[hash(comp_graph._nodes[idx])]["match"][::-1]:
                op_name, op_cost = item
                print(f"({op_name}, {op_cost:.2g})")
