# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Search for optimal fused sub-graphs and targets."""

import tvm
import numpy as np
from tvm._ffi.registry import register_func
import logging

make_tvm_fusion_spec = tvm._ffi.get_global_func("relay.collage.make_tvm_fusion_spec")
make_op_predicate_byoc_spec = tvm._ffi.get_global_func("relay.collage.make_op_predicate_byoc_spec")
make_labelled_dfpattern_fusion_rule = tvm._ffi.get_global_func("relay.collage.make_labelled_dfpattern_fusion_rule")
make_pattern_byoc_spec = tvm._ffi.get_global_func("relay.collage.make_pattern_byoc_spec")

def arg_for(type, device):
    """Returns a test argument for device of type"""
    return tvm.nd.array(np.random.uniform(-1.0, 1.0, size=type.concrete_shape).astype(type.dtype),
                        device=device)


@register_func("tvm.relay.collage.estimate_seconds")
def estimate_seconds(function, target):
    """Returns the mean execution time of the "Primitive" function on target.
       The function should already have any additional attributes needed to guide compilation,
       lowering and codegen."""
    device = tvm.device(target.kind.device_type)

    # The function will be marked "Primitive", and may optionally have a "Compiler" or other
    # annotation to guide lowering. Eta expand it so that we can compile a @main which will directly
    # invokes that compiled kernel.
    args = [arg_for(v.checked_type, device) for v in function.params]
    new_params = [tvm.relay.Var(v.name_hint, v.checked_type) for v in function.params]
    new_body = tvm.relay.Call(function, new_params)
    main = tvm.relay.Function(new_params, new_body)

    # Place that in a module and compile.
    mod = tvm.IRModule.from_expr(main)
    logging.info(f"Measuring overall module:\n{mod}")
    exe = tvm.relay.vm.compile(mod, target)

    # Benchmark the module.
    vm = tvm.runtime.vm.VirtualMachine(exe, device)
    benchmark_result = vm.benchmark(device, repeat=5, number=20, *args)

    logging.info(benchmark_result)
    return benchmark_result.mean  # seconds


@register_func("tvm.relay.collage.make_fusion_spec")
def make_fusion_spec(target):
    compiler = target.attrs.get("compiler", "")
    if compiler == "":
        logging.info(f"No 'compiler' attribute for target {target}, assuming built-in TVM lowering/codegen")
        return make_tvm_fusion_spec(target)
    else:
        pattern_table = tvm.relay.op.contrib.get_pattern_table(compiler)
        if pattern_table is None:
            logging.info(
                f"No pattern table for compiler {compiler} in {target}, assuming operator-predicate style BYOC lowering/codegen")
            return make_op_predicate_byoc_spec(target)
        else:
            logging.info(
                f"Converting {len(pattern_table)} rules for {compiler} in {target} for use in pattern style BYOC lowering/codegen")
            sub_rules = [make_labelled_dfpattern_fusion_rule(rule_name, dataflow_pattern, predicate) for
                         rule_name, dataflow_pattern, predicate in pattern_table]
            return make_pattern_byoc_spec(target, sub_rules)
