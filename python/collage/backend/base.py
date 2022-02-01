from enum import Enum
from collage.measurer.base import setup_mod_inputs
import tvm
from tvm import relay
from tvm.contrib import graph_executor as runtime
from collage.measurer.base import (
            NUM_MEASUREMENTS_PER_REPEAT, 
            NUM_REPEATS,
            measure,
        )
from collage.utils import (
            create_backend_pattern_annotation,
            get_backend_from_backend_pattern_annotation
        )


def default_cost_func(expr, codegen, target, **kwargs):
    # Prepare workload
    inputs = relay.analysis.free_vars(expr)
    expr_func = relay.Function(inputs, expr)
    net, params = relay.testing.create_workload(expr_func)

    # Generate executable
    lib = codegen(net, target, params, **kwargs)

    if isinstance(target, str):
        target = tvm.target.Target(target)
    dev = tvm.device(target.kind.device_type, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    setup_mod_inputs(module)
    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    # Measure
    return measure(ftimer, target)


# Special handling for op-level backends
# When BYOC pipeline is unified, this should be gone. 
def op_backend_cost_func(expr, codegen, target, **kwargs):
    # Prepare workload
    inputs = relay.analysis.free_vars(expr)
    expr_func = relay.Function(inputs, expr)

    assert("name" in kwargs)
    name = kwargs["name"]

    from ..optimizer.custom_fusion_pass import CustomFusionPass
    expr_func = expr_func.with_attr("CustomFusionPass", CustomFusionPass.OP_MEASUREMENT)
    default_op_group_id = 0
    annotation = create_backend_pattern_annotation(default_op_group_id, name)
    expr_func = expr_func.with_attr("BackendOP", annotation)
    expr_func = expr_func.with_attr("BuildTarget", target)
    net, params = relay.testing.create_workload(expr_func)

    kwargs["annotation"] = get_backend_from_backend_pattern_annotation(annotation)
    
    # Generate executable
    lib = codegen(net, target, params, **kwargs)

    if isinstance(target, str):
        target = tvm.target.Target(target)
    dev = tvm.device(target.kind.device_type, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    setup_mod_inputs(module)
    ftimer = module.module.time_evaluator("run", dev, number=NUM_MEASUREMENTS_PER_REPEAT, repeat=NUM_REPEATS)

    # Measure
    return measure(ftimer, target)


class BackendKind(Enum):
    OP_LEVEL = 1
    GRAPH_LEVEL = 2


# @sunggg
# Oftentimes, we want to reuse a pattern/pattern rule/pattern generator across multiple backends.
# Thus, we separate their object from backend and backend only has their pointer.
# e.g., conv2d+relu
class Backend:
    __lastId = 1
    def __init__(
                self, 
                name, 
                kind, 
                codegen, 
                patterns = None, 
                pattern_generator = None,
                cost_func = None,
                **kwargs
        ):

        self.id = Backend.__lastId
        Backend.__lastId += 1
        self.name = name
        assert(isinstance(kind, BackendKind))
        self.kind = kind
        self.codegen = codegen
        self.patterns = set() if patterns is None else set(patterns)
        self.pattern_generator = pattern_generator
        self.cost_func = default_cost_func if cost_func is None else cost_func
        self.kwargs = kwargs
        
    def measure_cost(self, expr, target, **kwargs):
        kwargs.update(self.kwargs)
        return self.cost_func(expr, self.codegen, target, **kwargs)

    def add_patterns(self, patterns):
        assert patterns is not None
        self.patterns.update(patterns)

    def add_pattern_generator(self, pattern_generator):
        assert pattern_generator is not None
        self.pattern_generator = pattern_generator

    def add_cost_func(self, cost_func):
        assert cost_func is not None
        self.cost_func = cost_func

       
