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

import tvm
import onnx
import numpy as np
import logging

from tvm.contrib.cutlass import build_cutlass_kernels_vm # force registration


logging.basicConfig(level=logging.INFO)

# rtx3070 is 8.6 but CUTLASS only supports 8.0
cutlass_sm = 80


def make_const_32(*shape):
    return tvm.relay.const(np.random.rand(*shape).astype("float32"))


def mnist():
    const0 = make_const_32(8, 1, 5, 5)
    const1 = make_const_32(8, 1, 1)
    const2 = make_const_32(16, 8, 5, 5)
    const3 = make_const_32(16, 1, 1)
    const4 = make_const_32(10, 256)
    const5 = make_const_32(1, 10)
    metatable = {"relay.Constant": [const0, const1, const2, const3, const4, const5]}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
          %0 = nn.pad(%x, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
          %1 = nn.conv2d(%0, meta[relay.Constant][0], padding=[0, 0, 0, 0], channels=8, kernel_size=[5, 5]);
          %2 = add(%1, meta[relay.Constant][1]);
          %3 = nn.relu(%2);
          %4 = nn.max_pool2d(%3, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0]);
          %5 = nn.pad(%4, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
          %6 = nn.conv2d(%5, meta[relay.Constant][2], padding=[0, 0, 0, 0], channels=16, kernel_size=[5, 5]);
          %7 = add(%6, meta[relay.Constant][3]);
          %8 = nn.relu(%7);
          %9 = nn.max_pool2d(%8, pool_size=[3, 3], strides=[3, 3], padding=[0, 0, 0, 0]);
          %10 = reshape(%9, newshape=[1, 256]);
          %11 = nn.dense(%10, meta[relay.Constant][4], units=None, out_dtype="float32");
          add(%11, meta[relay.Constant][5])
        }
        """,
        "from_string",
        None,
        metatable
    )
    return ({"x": [1, 1, 28, 28]}, {"x": "float32"}, mod, None)


def describe_onnx(model_file):
    """Returns the form of run to invoke ONNX model at model_file.
       Note that ? (ie unknown) shape dimensions must be changed to concrete dimensions
       which are consistent with the overall model."""
    onnx_model = onnx.load(model_file)
    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    for input_info in onnx_model.graph.input:
        if input_info.name not in initializer_names:
            _, shape, dtype, _ = tvm.relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise ValueError(
                    f"Unknown dtype on input '{input_info.name}' is not supported."
                )
            input_shapes.update({input_info.name: shape})
            input_dtypes.update({input_info.name: dtype})
    logging.info(f"input_shapes={input_shapes}")
    logging.info(f"input_dtypes={input_dtypes}")


def from_onnx(model_file, input_shapes, input_dtypes):
    logging.info("-------------------- BEGIN ONNX IMPORT --------------------")

    logging.info(f"Loading ONNX model from {model_file}")

    onnx_model = onnx.load(model_file)
    logging.info(f"Loaded model from {model_file}")

    mod, params = tvm.relay.frontend.from_onnx(onnx_model, input_shapes, freeze_params=True)
    mod = tvm.relay.transform.InferType()(mod)
    logging.info("-------------------- END ONNX IMPORT --------------------")

    logging.info(f"Imported model:\n{mod}")
    logging.info(f"Params:\n{params}")

    return input_shapes, input_dtypes, mod, params,


def compile_and_benchmark(input_shapes, input_dtypes, mod, params, targets, dev):
    exe = tvm.relay.vm.compile(mod, target=targets, params=params)
    vm = tvm.runtime.vm.VirtualMachine(exe, dev)
    args = {
        input_name: tvm.nd.array(
            np.random.random(input_shapes[input_name]).astype(input_dtypes[input_name]), device=dev
        )
        for input_name in input_shapes.keys()
    }
    profile_result = vm.benchmark(
        dev,
        func_name="main",
        number=1,
        repeat=1,
        min_repeat_ms=10,
        **args,
    ).results
    logging.info("time: {}ms".format(np.mean(profile_result) * 1e3))


def run(input_shapes, input_dtypes, mod, params):
    with tvm.transform.PassContext(config={"relay.fallback_device_type": 2, "relay.collage.enable_collage": True}):
        host_target = tvm.target.Target("llvm")
        generic_target = tvm.target.Target("cuda", host_target)
        cutlass_target = tvm.target.Target("cuda -compiler=cutlass", host_target)
        tensorrt_target = tvm.target.Target("cuda -compiler=tensorrt", host_target)
        targets = [generic_target, cutlass_target, tensorrt_target]
        dev = tvm.device(generic_target.kind.device_type)
        compile_and_benchmark(input_shapes, input_dtypes, mod, params, targets, dev)


def just_trt(input_shapes, input_dtypes, mod, params):
    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("cuda", host_target)
    targets = [generic_target]
    dev = tvm.device(generic_target.kind.device_type)
    mod, options = tvm.relay.op.contrib.partition_for_tensorrt(mod, params)
    logging.info("-------------- BEGIN PARTITIONED --------------")
    logging.info(mod)
    logging.info("-------------- END PARTITIONED ----------------")
    with tvm.transform.PassContext(config={"relay.ext.tensorrt.options": options}):
        compile_and_benchmark(input_shapes, input_dtypes, mod, params, targets, dev)


def just_cutlass(input_shapes, input_dtypes, mod, params):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        host_target = tvm.target.Target("llvm")
        generic_target = tvm.target.Target("cuda", host_target)
        targets = [generic_target]
        dev = tvm.device(generic_target.kind.device_type)
        mod = tvm.relay.op.contrib.partition_for_cutlass(mod, params)
        logging.info("-------------- BEGIN PARTITIONED --------------")
        logging.info(mod)
        logging.info("-------------- END PARTITIONED ----------------")
        compile_and_benchmark(input_shapes, input_dtypes, mod, params, targets, dev)


def just_tvm(input_shapes, input_dtypes, mod, params):
    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("cuda", host_target)
    targets = [generic_target]
    dev = tvm.device(generic_target.kind.device_type)
    logging.info("-------------- BEGIN MODULE --------------")
    logging.info(mod)
    logging.info("-------------- END MODULE ----------------")
    compile_and_benchmark(input_shapes, input_dtypes, mod, params, targets, dev)


if __name__ == "__main__":
    #just_trt(*from_onnx("/home/mbs/gauntlet/models/mnist-8.onnx", {"Input3": [1, 1, 28, 28]}, {"Input3": "float32"}))
    #just_cutlass(*mnist())
    run(*mnist())
