import taso as ts
import onnx

def get_pads(kernel, padding):
    if sum(padding) == 0 and sum(kernel) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    return pads

def make_conv2d(graph, input_tensor, filter_shape, strides, padding, actimode, name):
    kernel1, kernel2, in_dim, out_dim = filter_shape 
    kernel = (kernel1, kernel2)
    w = graph.new_weight(dims=(out_dim, in_dim, kernel1, kernel2))
    padding = get_pads(kernel, padding)
    t = graph.conv2d(input=input_tensor, weight=w, strides=strides, padding=padding, activation=actimode)
    return t 

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

def block(graph, tensor, inp, oup, stride, expand_ratio):
    convd = tensor

    hidden_dim = int(inp * expand_ratio)
    use_res_connect = stride == 1 and inp == oup

    #print(inp)
    if expand_ratio == 1:
        convd = make_conv2d(graph=graph, input_tensor=convd, filter_shape=(3,3,hidden_dim, hidden_dim), strides=(stride,stride), padding=(1,1), actimode="RELU", name="conv1")
        convd = make_conv2d(graph=graph, input_tensor=convd, filter_shape=(1,1,hidden_dim, oup), strides=(1,1), padding=(0,0), actimode="NONE", name="conv1")

        tensor = convd
    else:
        convd = make_conv2d(graph=graph, input_tensor=convd, filter_shape=(1,1,inp, hidden_dim), strides=(1,1), padding=(1,1), actimode="RELU", name="conv1")
        groups = hidden_dim
        #  nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        convd = make_conv2d(graph=graph, input_tensor=convd, filter_shape=(3,3,hidden_dim//groups, hidden_dim), strides=(stride,stride), padding=(1,1), actimode="RELU", name="conv1")
        convd = make_conv2d(graph=graph, input_tensor=convd, filter_shape=(1,1,hidden_dim, oup), strides=(1,1), padding=(0,0), actimode="NONE", name="conv1")

        tensor = convd
    
    if use_res_connect:
        tensor = graph.add(tensor, convd)

    return tensor

graph = ts.new_graph()
input = graph.new_input(dims=(1,32,56,56))

tensor = input
input_channel = 32
for t, c, n, s in inverted_residual_setting:
    output_channel = make_divisible(c * 1) if t > 1 else c
    for i in range(n):
        if i == 0:
            tensor = block(graph, tensor, input_channel, output_channel, s, expand_ratio=t)
        else:
            tensor = block(graph, tensor, input_channel, output_channel, 1, expand_ratio=t)
        input_channel = output_channel
tensor = make_conv2d(graph=graph, input_tensor=tensor, filter_shape=(1,1,input_channel,1280), strides=(1,1), padding=(0,0), actimode="RELU", name="last_conv")

new_graph = ts.optimize(graph, alpha=1.0, budget=100)
onnx_model = ts.export_onnx(new_graph)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "../taso-tvm/models/mobilenetv2_taso.onnx")

print(new_graph.run_time())
