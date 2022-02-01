import tvm
from tvm import relay
from tvm import topi
from tvm.relay.backend.te_compiler import LoweredOutput
from tvm.relay.op import op as _op
from tvm.relay.op.strategy.generic import wrap_topi_schedule, wrap_custom_compute_softmax, \
    wrap_custom_compute_activation, wrap_compute_conv3d, wrap_custom_compute_conv2d, wrap_custom_compute_pool2d, \
    wrap_custom_compute_biasadd, wrap_custom_compute_conv2d_add_relu, wrap_custom_compute_biasadd, \
    wrap_custom_compute_conv3d_add_relu, wrap_custom_compute_conv2d_relu, wrap_compute_dense, \
    wrap_compute_batch_matmul


def dummy():
    return None


@tvm._ffi.register_func("relay.backend.target_specific_lowering")
def target_specific_lowering(func, inputMap, target_info=None):
    strategy = _op.OpStrategy()

    calls = []

    def extract_attr(expr, calls):
        if type(expr) == tvm.relay.expr.Call:
            calls.append(expr)

    relay.analysis.post_order_visit(func, lambda expr: extract_attr(expr, calls))

    tokens = target_info.split('_')
    target = tokens[0]
    pattern = '_'.join(tokens[1:])

    def collect_input(inputMap):
        inputs = []
        for key, varray in inputMap.items():
            for val in varray:
                inputs.append(val)
        return inputs

    attrs, ret_type = None, None
    if target == "cudnn":
        # TODO: conv3d, avgpool, batchnorm
        if pattern == "0-Op(nn.softmax)[*]":
            strategy.add_implementation(
                wrap_custom_compute_softmax(topi.cuda.softmax_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="softmax.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(sigmoid)[*]":
            strategy.add_implementation(
                wrap_custom_compute_activation(topi.cuda.sigmoid_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="sigmoid.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(nn.relu)[*]":
            strategy.add_implementation(
                wrap_custom_compute_activation(topi.cuda.relu_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="relu.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(tanh)[*]":
            strategy.add_implementation(
                wrap_custom_compute_activation(topi.cuda.tanh_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="tanh.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        # TODO: not supported yet
        elif pattern == "0-Op(nn.bias_add)[*, *]":
            strategy.add_implementation(
                wrap_custom_compute_biasadd(topi.cuda.biasadd_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="biasadd.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(nn.conv2d)[*, *]":
            strategy.add_implementation(
                wrap_custom_compute_conv2d(
                    topi.cuda.conv2d_cudnn, need_data_layout=True, has_groups=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv2d.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(nn.conv3d)[*, *]":
            strategy.add_implementation(
                wrap_compute_conv3d(
                    topi.cuda.conv3d_cudnn, need_layout=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv3d.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)


        elif pattern == "0-Op(nn.max_pool2d)[*]":
            strategy.add_implementation(
                wrap_custom_compute_pool2d(topi.cuda.max_pool2d_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="max_pool2d.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(nn.avg_pool2d)[*]":
            strategy.add_implementation(
                wrap_custom_compute_pool2d(topi.cuda.avg_pool2d_cudnn),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="avg_pool2d.cudnn",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

            # TODO: not supported yet
            # elif pattern == "bn":
            # strategy.add_implementation(
            #    wrap_custom_compute_maxpool2d(topi.cuda.maxpool2d_cudnn),
            #    wrap_topi_schedule(topi.generic.schedule_extern),
            #    name="bn.cudnn",
            # )

            # has single op
            # attrs = calls[0].attrs
            eret_type = calls[0].checked_type
            # inputs = collect_input(inputMap)

        # fused ops
        elif pattern == "0-Op(nn.relu)[1-Op(add)[2-Op(nn.conv2d)[*, *], *]]":
            strategy.add_implementation(
                wrap_custom_compute_conv2d_add_relu(
                    topi.cuda.conv2d_add_relu_cudnn, need_data_layout=True, has_groups=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv2d+add+relu.cudnn",
            )

            data, kernel, Z, bias = None, None, None, None
            attrs, ret_type = None, None
            for call in calls:
                call_name = call.op.name
                if "conv2d" in call_name:
                    attrs = call.attrs
                    ret_type = call.checked_type
                    args = call.args
                    data = inputMap[args[0]]
                    kernel = inputMap[args[1]]
                elif "add" in call_name:
                    data2 = inputMap[args[1]]
                elif "relu" in call_name:
                    Z = inputMap[args[0]]

            inputs = [data[0], kernel[0], Z[0], data2[0]]

        elif pattern == "0-Op(nn.relu)[1-Op(nn.biad_add)[2-Op(nn.conv2d)[*, *], *]]":
            strategy.add_implementation(
                wrap_custom_compute_conv2d_add_relu(
                    topi.cuda.conv2d_bias_relu_cudnn, need_data_layout=True, has_groups=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv2d+bias+relu.cudnn",
            )

            data, kernel, Z, bias = None, None, None, None
            attrs, ret_type = None, None
            for call in calls:
                call_name = call.op.name
                if "conv2d" in call_name:
                    attrs = call.attrs
                    ret_type = call.checked_type
                    args = call.args
                    data = inputMap[args[0]]
                    kernel = inputMap[args[1]]
                elif "bias_add" in call_name:
                    data2 = inputMap[args[1]]
                elif "relu" in call_name:
                    Z = inputMap[args[0]]

            inputs = [data[0], kernel[0], Z[0], data2[0]]



        elif pattern == "0-Op(nn.relu)[1-Op(add)[2-Op(nn.conv3d)[*, *], *]]":
            strategy.add_implementation(
                wrap_custom_compute_conv3d_add_relu(
                    topi.cuda.conv3d_add_relu_cudnn, need_layout=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv3d+add+relu.cudnn",
            )

            data, kernel, Z, bias = None, None, None, None
            attrs, ret_type = None, None
            for call in calls:
                call_name = call.op.name
                if "conv3d" in call_name:
                    attrs = call.attrs
                    ret_type = call.checked_type
                    args = call.args
                    data = inputMap[args[0]]
                    kernel = inputMap[args[1]]
                elif "add" in call_name:
                    data2 = inputMap[args[1]]
                elif "relu" in call_name:
                    Z = inputMap[args[0]]

            inputs = [data[0], kernel[0], Z[0], data2[0]]



        elif pattern == "0-Op(nn.relu)[1-Op(nn.conv2d)[*, *]]":
            strategy.add_implementation(
                wrap_custom_compute_conv2d_relu(
                    topi.cuda.conv2d_relu_cudnn, need_data_layout=True, has_groups=True
                ),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="conv2d_relu.cudnn",
            )

            data, kernel, Z, bias = None, None, None, None
            attrs, ret_type = None, None
            for call in calls:
                call_name = call.op.name
                if "conv2d" in call_name:
                    attrs = call.attrs
                    ret_type = call.checked_type
                    args = call.args
                    data = inputMap[args[0]]
                    kernel = inputMap[args[1]]
                elif "add" in call_name:
                    bias = inputMap[args[1]]
                elif "relu" in call_name:
                    Z = inputMap[args[0]]

            inputs = [data[0], kernel[0]]

        else:
            # Unsupported backend op
            assert False, "{} is currently not supported in {}".format(target_info, target)


    # TODO: matmul vs dense?
    elif target == "cublas":
        if pattern == "0-Op(nn.dense)[*, *]":
            strategy.add_implementation(
                wrap_compute_dense(topi.cuda.dense_cublas),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="dense.cublas",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        elif pattern == "0-Op(nn.batch_matmul)[*, *]":
            strategy.add_implementation(
                wrap_compute_batch_matmul(topi.cuda.batch_matmul_cublas),
                wrap_topi_schedule(topi.generic.schedule_extern),
                name="batch_matmul.cublas",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        else:
            # Unsupported backend op
            assert False, "{} is currently not supported in {}".format(target_info, target)

    elif target == "mkl":
        if pattern == "0-Op(nn.dense)[*, *]":
            from tvm.te import SpecializedCondition
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

            same_type = inputs[0].dtype == inputs[1].dtype == ret_type.dtype
            dtype = inputs[0].dtype

            with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
                strategy.add_implementation(
                    wrap_compute_dense(topi.x86.dense_mkl),
                    wrap_topi_schedule(topi.x86.schedule_dense_mkl),
                    name="dense.mkl",
                )
        elif pattern == "0-Op(nn.batch_matmul)[*, *]":
            strategy.add_implementation(
                wrap_compute_batch_matmul(topi.x86.batch_matmul_mkl),
                wrap_topi_schedule(topi.x86.schedule_batch_matmul_mkl),
                name="batch_matmul.mkl",
            )
            # has single op
            attrs = calls[0].attrs
            ret_type = calls[0].checked_type
            inputs = collect_input(inputMap)

        else:
            # Unsupported backend op
            assert False, "{} is currently not supported in {}".format(target_info, target)

    elif target == "tensorrt":
        assert False, f"{target} should be passed to the external compiler"

    elif target == "dnnl":
        assert False, f"{target} should be passed to the external compiler"

    else:
        assert False, f"Unsupported target: {target}"

    # To compute subgraph
    #   attrs for each op
    #   input for the subgraph
    #   -  pattern - will be given

    #  May need rewrite?
    #

    for spec in strategy.specializations:
        # if spec.condition:
        for impl in spec.implementations:
            # attribute, inputs, output_type
            outputs = impl.compute(attrs, inputs, ret_type)
            return LoweredOutput(outputs, impl)

    # Should not reach
    return None
