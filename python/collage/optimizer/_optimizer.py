import tvm._ffi
import tvm.driver
from collage.pattern_manager.pattern_registry import PatternRegistry
from collage.utils import get_function_body, is_constant_node
from .comp_graph import ComputationGraph
from .comp_graph_optimizer import (
    CompGraphOptimizer,
    AssignBackendExprVisitor,
)
from .ext_compiler_op_annotator import ExtCompilerOpAnnotator
from tvm.relay.op.contrib.tensorrt import prune_tensorrt_subgraphs
from tvm.relay import transform

import logging
from collage.interface import CollageContext
from .op_match_logger import OpMatchLogger, OpMatchReader
from collage.backend import BackendKind


def setup_pattern_registry(hw_name):
    pattern_registry = PatternRegistry.get(hw_name)
    return pattern_registry


@tvm._ffi.register_func("collage.optimizer.print_attr_args")
def print_attr_args(expr):
    logger.info(f"attr: {get_attr_vals(expr)}")


@tvm._ffi.register_func("collage.optimizer.visualize_network_debug")
def visualize_network_debug(relay_expr, name):
    pass


def apply_tensorrt_op(mod):
    logging.info("Applying TensorRT op pass")
    from tvm.relay.op.contrib.tensorrt import RemoveDropoutPass
    # Get best op match info
    fn_body = mod["main"].body
    # Annotating expression
    target_str = "tensorrt"

    # Do merge and partition pass
    use_implicit_batch = False
    remove_no_mac_subgraphs = False
    max_workspace_size = 1 << 30
    version = None

    config = {
        "use_implicit_batch": use_implicit_batch,
        "max_workspace_size": max_workspace_size,
        "remove_no_mac_subgraphs": remove_no_mac_subgraphs,
    }

    if version:
        assert isinstance(version, tuple) and len(version) == 3
        config["tensorrt_version"] = version
    else:
        linked_version = tuple(tvm.get_global_func("relay.op.get_tensorrt_version")())
        if not linked_version:
            logger.warning(
                "TVM was not built against TensorRT and no version was provided to "
                "partition_for_tensorrt. Defaulting to 6.0.1"
            )
            linked_version = (6, 0, 1)
        config["tensorrt_version"] = linked_version

    # Warning(@Soo): I assume this is only useful when folding constant
    seq = tvm.transform.Sequential([
        transform.InferType(),
        transform.FoldConstant(),
        transform.MergeComposite(tvm.relay.op.contrib.get_pattern_table("tensorrt"), "tensorrt"),
        transform.AnnotateTarget("tensorrt"),
        transform.MergeCompilerRegions(),
        transform.PartitionGraph(),
        transform.InferType()])

    # Do prune_tensorrt_subgraphs
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
        mod = seq(mod)
    return mod


def apply_dnnl_op(mod):
    opt_pass = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = opt_pass(mod)

    return mod


@tvm._ffi.register_func("collage.optimizer.apply_external_compiler_op")
def apply_external_compiler_op(mod):
    target = mod["main"].attrs["BuildTarget"]
    if "cuda" in target.keys:
        mod = apply_tensorrt_op(mod)
    elif "llvm" in target.keys:
        mod = apply_dnnl_op(mod)
    else:
        Exception(f"Unexpected HW for external compiler op pass: {target.keys}")

    return mod
    # return mod, config


@tvm._ffi.register_func("collage.optimizer.get_user_fusion")
def get_user_fusion(relay_expr):
    logging.info("User-defined fusion")
    relay_expr = get_function_body(relay_expr)
    match_path = CollageContext.graph_level_tmp_file
    opt_match = OpMatchReader().read(relay_expr, match_path)


@tvm._ffi.register_func("collage.optimizer.visualize_backend_placement")
def run_backend_placement_visualization(relay_expr):
    pass


def get_backends(func_expr, backend_registry):
    assert ("BackendList" in func_expr.attrs)
    backend_list_str = func_expr.attrs["BackendList"]
    backend_str_list = backend_list_str.split(",")
    backends = [backend_registry[b] for b in backend_str_list]

    return backends


def get_backend_names(backends):
    return [b.name for b in backends]


def run_op_level_opt(func_expr):
    target = func_expr.attrs["BuildTarget"]
    pattern_registry = CollageContext.pattern_registry
    backend_registry = pattern_registry.backend_registry
    given_backends = get_backends(func_expr, backend_registry)
    relay_expr = get_function_body(func_expr)

    logging.info(f"[Op-Level: DP] Computation graph generation...")
    comp_graph = ComputationGraph(relay_expr)
    n_relay_nodes = comp_graph.n_relay_nodes
    logging.info(f"# of relay nodes in comp graph: {n_relay_nodes}")

    # Optimizing graph

    assert (pattern_registry is not None)
    optimizer = CompGraphOptimizer(pattern_registry, given_backends)

    """
    Warning(@Soo): Note that current DP optimizer does not work for patterns with more than one root.
    For example, Conv     Conv (Two parallel convolution) case can't be handled
                   \       /
                      ReLU
    Following lines need to be modified to afford more than two roots
    - pat.get_relay_pattern().match(f_expr)

    This is because of inherent limitation of Relay pattern and
    the discrepancy between what Relay pattern supports and how TVM fusion strategy works.
    We can come back to this later if this is critical to performance, which is unlikely for now given networks we have.
    """
    optimized_match = optimizer.optimize(comp_graph, target)

    logging.info(
        "[Op-Level: DP] It finished optimizing comp graph and assigning backend ops to Relay Expr (backend attr)")

    # Save fisrt layer best results
    OpMatchLogger().save(relay_expr, optimized_match, log_path=CollageContext.op_level_placement_log)
    return optimized_match, relay_expr, pattern_registry, n_relay_nodes


@tvm._ffi.register_func("collage.optimizer.run_dp")
def run_dp(relay_expr):
    run_op_level_opt(relay_expr)


@tvm._ffi.register_func("collage.optimizer.assign_backend_for_op_measurement")
def assign_backend_for_op_measurement(relay_expr):
    backend_pattern_name = relay_expr.attrs["BackendOP"]
    assert isinstance(backend_pattern_name, str)

    relay_expr = get_function_body(relay_expr)
    AssignBackendExprVisitor().assign(relay_expr, backend_pattern_name)

    # logger.info(repr(relay_expr))
    # sys.exit(0)
