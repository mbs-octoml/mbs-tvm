import tvm
import os
from tvm import relay, autotvm
from collage.pattern_manager.pattern_registry import PatternRegistry
from collage.measurer.op_cost_logger import OpCostLogger
from collage.backend import (Backend, BackendKind)

from collage.pattern_manager.default_patterns import (
                    cudnn_default_patterns,
                    cublas_default_patterns,
                    mkl_default_patterns,
                    dnnl_default_patterns,
                    cutlass_default_patterns,
                )
from collage.pattern_manager.default_pattern_rules import (
                    DefaultPatternGenerator,
                    tvm_pattern_generator,
                    trt_pattern_generator
                )
from collage.backend.base import op_backend_cost_func
from collage.backend.default_backends import (
                    cg_AutoTVM,
                    cg_TensorRT,
                    cg_cuDNN,
                    cg_cuBLAS,
                    cg_MKL,
                    cg_DNNL,
                    cg_CUTLASS,
                )
import logging

def _register_new_backend(
            registry,
            name,
            kind,
            codegen,
            patterns = None,
            pattern_generator = None,
            cost_func = None,
            **kwargs
        ):

    assert(name not in registry)
    registry[name] = Backend(
            name = name,
            kind = kind,
            codegen = codegen,
            patterns = patterns,
            pattern_generator = pattern_generator,
            cost_func = cost_func,
            **kwargs
        )


def _register_default_backends(registry):
    _register_new_backend(registry,
                name = "autotvm",
                kind = BackendKind.OP_LEVEL,
                codegen = cg_AutoTVM,
                pattern_generator = tvm_pattern_generator, # valid_op + fusion_rule
                tuning_log=f"autotvm_tuning_log.json"
            )

    _register_new_backend(registry,
                name = "cudnn",
                kind = BackendKind.OP_LEVEL,
                codegen = cg_cuDNN,
                patterns = cudnn_default_patterns,
                cost_func = op_backend_cost_func,
            )

    _register_new_backend(registry,
                name = "cublas",
                kind = BackendKind.OP_LEVEL,
                codegen = cg_cuBLAS,
                patterns = cublas_default_patterns,
                cost_func = op_backend_cost_func,
            )

    _register_new_backend(registry,
                name = "tensorrt",
                kind = BackendKind.GRAPH_LEVEL,
                codegen = cg_TensorRT,
                pattern_generator = trt_pattern_generator,
            )

    _register_new_backend(registry,
                name = "mkl",
                kind = BackendKind.OP_LEVEL,
                codegen = cg_MKL,
                patterns = mkl_default_patterns,
                cost_func = op_backend_cost_func,
            )

    _register_new_backend(registry,
                name = "dnnl",
                kind = BackendKind.GRAPH_LEVEL,
                codegen = cg_DNNL,
                patterns = dnnl_default_patterns,
            )

    _register_new_backend(registry,
                name = "cutlass",
                kind = BackendKind.GRAPH_LEVEL,
                codegen = cg_CUTLASS,
                patterns = cutlass_default_patterns,
            )


this_code_path = os.path.dirname(os.path.abspath(__file__))

class CollageContext:
    backends = None
    pattern_registry = None
    op_cost_logger = None
    op_level_placement_log = None
    graph_level_placement_log = None
    graph_level_tmp_file = None
    evolutionary_search_pop_size = 0
    evolutionary_search_max_iter = 0

    def __init__(
                self,
                mod,
                backends,
                op_level_placement_log = "op_level_placement.log",
                graph_level_placement_log = "graph_level_placement.log",
                graph_level_tmp_file = "graph_lv.tmp",
                ev_pop_size = 50,
                ev_max_iter = 100000,
                search_budget = 0.5,
                input_placement_log_file = None,
                placement_vis_file = None
            ):
        CollageContext.pattern_registry = mod.pattern_registry
        CollageContext.op_cost_logger = mod.op_cost_logger
        CollageContext.backends = backends
        CollageContext.op_level_placement_log = op_level_placement_log
        CollageContext.graph_level_placement_log = graph_level_placement_log
        CollageContext.graph_level_tmp_file = graph_level_tmp_file
        CollageContext.evolutionary_search_pop_size = ev_pop_size
        CollageContext.evolutionary_search_max_iter = ev_max_iter
        CollageContext.evolutionary_search_budget = search_budget

        # Arguments for backend placement visualization
        CollageContext.input_placement_log_file = input_placement_log_file
        CollageContext.placement_vis_file = placement_vis_file

    def __enter__(self):
        logging.info("Entering Collage")
        pass

    def __exit__(self, exc_type, exc_value, tb):
        logging.info("Exiting Collage")
        pass

def get_absolute_path(path):
    if not os.path.isabs(path):
        path = os.path.abspath(os.getcwd()) + "/" + path
    return path
        


class Module:
    def __init__(
                  self,
                  op_cost_log_path = None,
                  dump_readable_cost_log  = False,
                  op_level_placement_log_path= None,
                  graph_level_placement_log_path = None,
                  graph_level_tmp_file_path = None
                ):
        backend_registry = dict()
        _register_default_backends(backend_registry)
        self.op_cost_logger = OpCostLogger(op_cost_log_path, dump_readable_cost_log)
        self.op_cost_logger.load_from_log()

        self.pattern_registry = PatternRegistry(
                        backend_registry,
                        self.op_cost_logger,
                    )
        op_level_placement_log_path = "op_level_placement.log" if op_level_placement_log_path is None else op_level_placement_log_path
        graph_level_placement_log_path = "graph_level_placement.log" if graph_level_placement_log_path is None else graph_level_placement_log_path
        graph_level_tmp_file_path = "/tmp/graph_lv.tmp" if graph_level_tmp_file_path is None else graph_level_tmp_file_path
        
        self.op_level_placement_log = get_absolute_path(op_level_placement_log_path)
        self.graph_level_placement_log = get_absolute_path(graph_level_placement_log_path)
        self.graph_level_tmp_file = get_absolute_path(graph_level_tmp_file_path)


    def register_new_backend(
                                self, 
                                name, 
                                kind, 
                                codegen, 
                                patterns = None,
                                pattern_generator = None,
                                cost_func = None,
                                **kwargs
                            ):
        _register_new_backend(
                                self.pattern_registry.backend_registry, 
                                name, 
                                kind, 
                                codegen, 
                                patterns,
                                pattern_generator,
                                cost_func,
                                **kwargs
                            )

    def update_existing_backend(self, name, kind, codegen, **kwargs):
        assert(name in registry)
        registry[name] = registry[name].kind = kind
        registry[name] = registry[name].codegen = codegen
        registry[name] = registry[name].kwargs = kwargs

    def get_registered_backends(self):
        return self.pattern_registry.get_registered_backends()

    def add_backend_patterns(self, backend_name, patterns):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        backend_registry[backend_name].add_patterns(patterns)

    # This will use base pattern generator
    def add_backend_pattern_rule(self, backend_name, pattern_rule):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        backend_registry[backend_name].pattern_generator = DefaultPatternGenerator(pattern_rule)

    def add_backend_pattern_generator(self, backend_name, pattern_generator):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        backend_registry[backend_name].pattern_generator = pattern_generator

    def add_backend_pattern_generator(self, backend_name, cost_func):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        backend_registry[backend_name].cost_func = cost_func

    def get_backend_patterns(self, backend_name):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        return list(backend_registry[backend_name].patterns)

    def get_backend_pattern_generator(self, backend_name):
        backend_registry = self.pattern_registry.backend_registry
        assert(backend_name in backend_registry)
        return backend_registry[backend_name].pattern_generator

    def update_backend_tuning_log(self, backend_name, log_path):
        self.pattern_registry.backend_registry[backend_name].kwargs["tuning_log"] = log_path

    def optimize_backend_placement(
                                    self,
                                    optimizer,
                                    backends,
                                    network_name,
                                    mod,
                                    params,
                                    target,
                                    batch_size,
                                    **kwargs
                                ):

        self.pattern_registry.update_backend_registry_info()
        net = mod["main"]
        from collage.optimizer.custom_fusion_pass import CustomFusionPass
        assert(optimizer == "op-level" or optimizer == "two-level")
        optimizer = CustomFusionPass.DP if optimizer == "op-level" else CustomFusionPass.TWO_LEVEL_OPT

        net = net.with_attr("CustomFusionPass", optimizer)
        net = net.with_attr("BuildTarget", target)

        # We need to pass these for now.
        net = net.with_attr("Network", network_name)
        net = net.with_attr("BatchSize", batch_size)
        net = net.with_attr("BackendList", ",".join(backends))

        

        ev_pop_size = kwargs["ev_pop_size"] if "ev_pop_size" in kwargs else 50
        ev_max_iter = kwargs["ev_max_iter"] if "ev_pop_size" in kwargs else 100000
        ev_budget = kwargs["ev_budget"] if "ev_budget" in kwargs else 0.3

        autotvm_tuning_log = self.pattern_registry.backend_registry["autotvm"].kwargs["tuning_log"]
        # Optimize
        with CollageContext(
                    self, 
                    backends, 
                    self.op_level_placement_log,
                    self.graph_level_placement_log, 
                    self.graph_level_tmp_file, 
                    ev_pop_size, 
                    ev_max_iter,
                    ev_budget
                ):
            with autotvm.apply_history_best(autotvm_tuning_log):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(net, target, params=params)

        return lib

    def visualize_backend_placement(
                                    self,
                                    backends,
                                    network_name,
                                    mod,
                                    params,
                                    target,
                                    batch_size,
                                    input_placement_log_file,
                                    placement_vis_file,
                                    **kwargs
                                ):
        net = mod["main"]
        from collage.optimizer.custom_fusion_pass import CustomFusionPass
        net = net.with_attr("CustomFusionPass", CustomFusionPass.VISUALIZE_BACKEND_PLACEMENT)
        net = net.with_attr("BuildTarget", target)

        # We need to pass these for now.
        net = net.with_attr("Network", network_name)
        net = net.with_attr("BatchSize", batch_size)
        net = net.with_attr("BackendList", ",".join(backends))

        autotvm_tuning_log = self.pattern_registry.backend_registry["autotvm"].kwargs["tuning_log"]

        ev_pop_size = kwargs["ev_pop_size"] if "ev_pop_size" in kwargs else 50
        ev_max_iter = kwargs["ev_max_iter"] if "ev_pop_size" in kwargs else 100000
        ev_budget = kwargs["ev_budget"] if "ev_budget" in kwargs else 0.3

        # Optimize
        with CollageContext(
                    self,
                    backends,
                    self.op_level_placement_log,
                    self.graph_level_placement_log,
                    self.graph_level_tmp_file,
                    ev_pop_size,
                    ev_max_iter,
                    ev_budget,
                    get_absolute_path(input_placement_log_file),
                    get_absolute_path(placement_vis_file)
                ):
            with autotvm.apply_history_best(autotvm_tuning_log):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(net, target, params=params)

