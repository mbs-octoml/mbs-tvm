import tvm
from tvm import relay
# import tensorflow as tf
# import tvm.relay.testing.tf as tf_testing
# import tvm.relay.testing as testing
from .utils import create_relay_workload

# try:
#     tf_compat_v1 = tf.compat.v1
# except ImportError:
#     tf_compat_v1 = tf

def get_conv2d(batch_size):
    # Parameters
    out_channels = 16
    # batch_size = 1

    # Network definition
    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
    expr = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )

    # Workload
    mod, params = create_relay_workload(expr)
    shape_dict = {
        "data": (batch_size, 3, 224, 224),
        # "weight": (16, 3, 3, 3)
    }

    return mod, params, shape_dict

def get_conv2d_relu(batch_size):
    # Chain graph
    out_channels = 16
    # batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
    expr = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    expr = relay.nn.relu(expr)

    mod, params = create_relay_workload(expr)
    shape_dict = {
        "data": (batch_size, 3, 224, 224),
        "weight": (16, 3, 3, 3)
    }

    return mod, params, shape_dict

def get_diamond(batch_size):
    # Chain graph
    out_channels = 16
    # batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 64, 224, 224), "float32"))
    # data2 = relay.var("data2", relay.TensorType((batch_size, 16, 224, 224), "float32"))
    conv_weight = relay.var("weight", relay.TensorType((out_channels, 64, 3, 3), "float32"))

    #conv_weight2 = relay.var("weight2", relay.TensorType((out_channels, out_channels, 3, 3), "float32"))
    #conv_weight3 = relay.var("weight3", relay.TensorType((out_channels, out_channels, 3, 3), "float32"))
    expr = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    path1 = relay.nn.relu(expr)
    #path1 = relay.nn.conv2d(
    #    data=path1, weight=conv_weight2, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    #)
    #path2 = relay.nn.conv2d(
    #    data=expr, weight=conv_weight3, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    #)
    path2 = relay.tanh(expr)
    path2 = relay.sigmoid(path2)
    # path2 = relay.add(path2, data2)
    expr = relay.add(path1, path2)

    mod, params = create_relay_workload(expr)

    shape_dict = {
        "data": (batch_size, 64, 224, 224),
        # "data2": (batch_size, 16, 224, 224),
        # "weight": (64, 64, 3, 3),
    }

    return mod, params, shape_dict


# This is first two conv + relu in ResNet-50
def get_conv2d_relu_x2(batch_size):
    # Chain graph
    # batch_size = 1

    data = relay.var("data", relay.TensorType((batch_size, 64, 56, 56), "float32"))
    conv_weight = relay.var("2_weight", relay.TensorType((64, 64, 1, 1), "float32"))
    expr = relay.nn.conv2d(
        data=data, weight=conv_weight, kernel_size=(1, 1), channels=64, padding=(0, 0)
    )
    expr = relay.nn.relu(expr)

    # Input size: [1, 64, 58, 58]
    # expr = relay.var("data", relay.TensorType((batch_size, 64, 58, 58), "float32"))
    conv_weight2 = relay.var("1_weight", relay.TensorType((64, 64, 3, 3), "float32"))
    expr = relay.nn.conv2d(
        data=expr, weight=conv_weight2, kernel_size=(3, 3), channels=64, padding=(1, 1)
    )
    expr = relay.nn.relu(expr)

    mod, params = create_relay_workload(expr)
    mod = relay.transform.InferType()(mod)
    # print(mod["main"].body.checked_type)

    shape_dict = {
        "data": (batch_size, 64, 56, 56),
        # "2_weight": (64, 64, 1, 1),
        # "1_weight": (64, 64, 3, 3)
    }

    return mod, params, shape_dict

def get_annotate_test(batch_size):
    data = relay.var("data", shape=(10, 10))
    O_1 = relay.nn.relu(data)
    O_2 = relay.nn.relu(O_1)
    X = relay.tanh(O_1)
    O_3 = relay.add(O_2, X)
    # diamond = relay.Function([data], O_3)

    mod, params = create_relay_workload(O_3)
    mod = relay.transform.InferType()(mod)
    return mod, params

def get_reshape(batch_size):
    # Network definition
    data = relay.var("data", relay.TensorType((8, 64, 32), "float32"))
    expr = relay.reshape(data=data, newshape=(1,8,64,32))

    # Workload
    mod, params = create_relay_workload(expr)
    shape_dict = {
        "data": (8, 64, 32),
    }

    return mod, params, shape_dict

# def get_conv2d_relu_x2(batch_size):
#     # Chain graph
#     # batch_size = 1
#
#     data = relay.var("data", relay.TensorType((batch_size, 64, 56, 56), "float32"))
#     conv_weight = relay.var("weight", relay.TensorType((64, 64, 1, 1), "float32"))
#     expr = relay.nn.conv2d(
#         data=data, weight=conv_weight, kernel_size=(1, 1), channels=64, padding=(1, 1)
#     )
#     expr = relay.nn.relu(expr)
#
#     # Input size: [1, 64, 58, 58]
#     expr = relay.nn.conv2d(
#         data=expr, weight=conv_weight, kernel_size=(1, 1), channels=64, padding=(1, 1)
#     )
#     expr = relay.nn.relu(expr)
#
#     mod, params = create_relay_workload(expr)
#     # mod = relay.transform.InferType()(mod)
#     # print(mod["main"].body.checked_type)
#
#     return mod, params


# def get_batch_matmul():
#     data1 = relay.var("data", relay.TensorType((16, 16, 1024), "float32"))
#     data2 = relay.var("data", relay.TensorType((16, 16, 1024), "float32"))
#     batmul = relay.nn.batch_matmul(data1, data2)
#
#     inputs = relay.analysis.free_vars(batmul)
#     expr_func = relay.Function(inputs, batmul)
#     net, params = testing.create_workload(expr_func)
#
#     return net, params
#
#
# def get_concat():
#     data1 = relay.var("data1", relay.TensorType((1, 3, 224, 224), "float32"))
#     data2 = relay.var("data2", relay.TensorType((1, 3, 224, 224), "float32"))
#     data3 = relay.var("data3", relay.TensorType((1, 3, 224, 224), "float32"))
#     data4 = relay.var("data4", relay.TensorType((1, 3, 224, 224), "float32"))
#     tup = relay.Tuple([data1, data2, data3, data4])
#     return tup
#     #concat = relay.concatenate(tup, axis=1)
#     #return concat
#
# def get_chain_graph():
#     # Chain graph
#     out_channels = 16
#     batch_size = 1
#
#     data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
#     conv_weight = relay.var("weight", relay.TensorType((out_channels, 3, 3, 3), "float32"))
#     dense_weight = relay.var("weight", relay.TensorType((10, 802816), "float32"))
#     # bn_gamma = relay.var("bn_gamma")
#     # bn_beta = relay.var("bn_beta")
#     # bn_mmean = relay.var("bn_mean")
#     # bn_mvar = relay.var("bn_var")
#
#     # simple_net = relay.nn.relu(data)
#     # simple_net = relay.nn.relu(simple_net)
#     simple_net = relay.nn.conv2d(
#         data=data, weight=conv_weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
#     )
#     simple_net = relay.nn.relu(simple_net)
#
#     # simple_net = relay.nn.batch_flatten(simple_net)
#     # simple_net = relay.nn.dense(simple_net, dense_weight)
#     # simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
#     # simple_net = relay.nn.relu(simple_net)
#
#     expr = simple_net
#
#     inputs = relay.analysis.free_vars(expr)
#     expr_func = relay.Function(inputs, expr)
#     net, params = testing.create_workload(expr_func)
#
#     return net, params
#
# def get_resnet_8():
#     batch_size = 1
#     num_class = 1000
#     # Resnet-8
#     image_shape = (3, 28, 28)
#
#     # ResNet-2
#     # mod, params = testing.resnet.get_workload(num_layers=2, batch_size=batch_size, image_shape=image_shape)
#
#     # Resnet-8
#     mod, params = testing.resnet.get_workload(num_layers=8, batch_size=batch_size, image_shape=image_shape)
#     # expr_func = mod["main"]
#     # net, params = testing.create_workload(expr_func)
#
#     return mod, params


"""
Three ways to create a Relay workload
--------------
1) Define the network in relay frontend API.
2) load some pre-defined network from :code:`tvm.relay.testing`.
3) load models from MXNet, ONNX and TensorFlow.

We only enable the first method for now. (@Soo)

"""

NAME_TO_WORKLOAD = {
    "conv2d+relu":get_conv2d_relu,
    "conv2d":get_conv2d,
    "conv2d+relu_x2":get_conv2d_relu_x2,
    "diamond":get_diamond,
    "annotate_test":get_annotate_test,
    "reshape":get_reshape,
}

def get_network_from_relay(name, batch_size):
    func = None
    if name in NAME_TO_WORKLOAD:
        mod, params, shape_dict = NAME_TO_WORKLOAD[name](batch_size)
    else:
        raise ValueError("Unsupported workload: " + name)

    return mod, params, shape_dict, None

    # """Get the symbol definition and random weight of a network"""
    # dtype = 'float32'
    # input_shape = (batch_size, 3, 224, 224)
    # output_shape = (batch_size, 1000)
    #
    # if "resnet" in name:
    #     n_layer = int(name.split("-")[1])
    #     mod, params = relay.testing.resnet.get_workload(
    #         num_layers=n_layer, batch_size=batch_size, dtype=dtype
    #     )
    # elif "vgg" in name:
    #     n_layer = int(name.split("-")[1])
    #     mod, params = relay.testing.vgg.get_workload(
    #         num_layers=n_layer, batch_size=batch_size, dtype=dtype
    #     )
    # elif name == "mobilenet":
    #     mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    # elif name == "squeezenet_v1.1":
    #     mod, params = relay.testing.squeezenet.get_workload(
    #         batch_size=batch_size, version="1.1", dtype=dtype
    #     )
    # elif name == "inception_v3":
    #     input_shape = (batch_size, 3, 299, 299)
    #     mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    # elif name == "mxnet":
    #     # an example for mxnet model
    #     from mxnet.gluon.model_zoo.vision import get_model
    #
    #     block = get_model("resnet18_v1", pretrained=True)
    #     mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
    #     net = mod["main"]
    #     net = relay.Function(
    #         net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    #     )
    #     mod = tvm.IRModule.from_expr(net)
    # elif name == "bert":
    #     # load bert protobuf
    #
    #     # sym, params = relay.frontend.from_tensorflow(output_nodes)
    #     #config = tf.ConfigProto()
    #     #with tf.Session(config=config) as sess:
    #     model_path = '../autotune-rtx2070/tf_models/models_pb/bert'
    #
    #     with tf_compat_v1.Session() as sess:
    #
    #         meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], model_path)
    #         graph_def = meta_graph_def.graph_def
    #         graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    #
    #         # Bert from TASO doesn't include softmax
    #         # graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")
    #
    #     layout = None
    #     input_shape = (64, 1024)
    #     shape_dict = {"DecodeJpeg/contents": input_shape}
    #     mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
    #     print("Compelete")

    # return mod, params#, input_shape, output_shape
