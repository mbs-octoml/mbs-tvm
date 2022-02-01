from tvm import relay
import tvm.relay.testing as testing
#from collage.pattern_manager.utils import *

#######################################################################
# Warning(@Soo): Deprecated; It turns out that we need to run model code as a main (e.g., bert.py)
# not to get an error for BERT and DCGAN.
#######################################################################

# TF2 model loading
# from baselines.tf2.bert import bert_tf2, bert_tf2_xla
# from baselines.tf2.mobilenetv2 import mobilenetv2_tf2, mobilenetv2_tf2_xla
# from baselines.tf2.nasrnn import nasrnn_tf2, nasrnn_tf2_xla
# from baselines.tf2.nasnet_a import nasneta_tf2, nasneta_tf2_xla
# from baselines.tf2.resnet50 import resnet50_tf2, resnet50_tf2_xla
# from baselines.tf2.resnext50 import resnext50_tf2, resnext50_tf2_xla
# from baselines.tf2.resnet50_3d import resnet50_3d_tf2, resnet50_3d_tf2_xla
# from baselines.tf2.dcgan import dcgan_tf2, dcgan_tf2_xla

# TF2 Model
# NETWORK_TO_TF2_MODEL = {
#     "bert": bert_tf2,
#     "bert_xla": bert_tf2_xla,
#
#     "mobilenet_v2": mobilenetv2_tf2,
#     "mobilenet_v2_xla": mobilenetv2_tf2_xla,
#
#     "nasrnn": nasrnn_tf2,
#     "nasrnn_xla": nasrnn_tf2_xla,
#
#     "nasneta": nasneta_tf2,
#     "nasneta_xla": nasneta_tf2_xla,
#
#     "resnet50": resnet50_tf2,
#     "resnet50_xla": resnet50_tf2_xla,
#
#     "resnext50_32x4d": resnext50_tf2,
#     "resnext50_32x4d_xla": resnext50_tf2_xla,
#
#     "resnet50_3d": resnet50_3d_tf2,
#     "resnet50_3d_xla": resnet50_3d_tf2_xla,
#
#     "dcgan": dcgan_tf2,
#     "dcgan_xla": dcgan_tf2_xla,
# }

# Key is network name and batch size
# Value is shape dict
WORKLOADS_DIC = {
    # Validated for ONNX
    "mnist-8": {1: {"Input3":[1, 1, 28, 28]}},
    "gpt2": {1: {"input1": [1, 8, 8]}},

    # Original
    "resnet_block": {1: {"input0": [1, 64, 56, 56]},
                     8: {"input0": [8, 64, 56, 56]},
                     16: {"input0": [16, 64, 56, 56]}},
    "resnet50": {1: {"input0": [1, 64, 56, 56]},
                 8: {"input0": [8, 64, 56, 56]},
                 16: {"input0": [16, 64, 56, 56]}},
    "resnext50_32x4d": {1: {"input0": [1, 64, 56, 56]},
                        4: {"input0": [4, 64, 56, 56]},
                        8: {"input0": [8, 64, 56, 56]},
                        16: {"input0": [16, 64, 56, 56]}},
    "nasneta": {1: {"input0": [1, 64, 56, 56]},
                8: {"input0": [8, 64, 56, 56]},
                16: {"input0": [16, 64, 56, 56]}},
    # NasRNN always have some errors during autotuning operators with AutoTVM
    # "nasrnn": {'x.1': [1, 512]},
    # "nasrnn": {'x.1': [1, 1024]},
    # "nasrnn": {'x.1': [1, 2048]},
    "nasrnn": {1: {'x.1': [1, 2560]}},
    # "nasrnn": {'x.1': [1, 512], 'x.2': [1, 512], 'x.3': [1, 512], 'x.4': [1, 512], 'x': [1, 512]},
    "bert": {1: {"input0": [64, 1024]}},
    "bert_full": {1: {"input0": [1, 64, 256]},  # (batch_size, max_seq_len, n_hidden)
                  8: {"input0": [8, 64, 256]},
                  16: {"input0": [16, 64, 256]}},
    "resnet50_3d": {1: {"input0": [1, 64, 3, 56, 56]},
                    8: {"input0": [8, 64, 3, 56, 56]},
                    16: {"input0": [16, 64, 3, 56, 56]}},
    "mobilenet_v2": {1: {"input0": [1, 32, 224, 224]},
                     8: {"input0": [8, 32, 224, 224]},
                     16: {"input0": [16, 32, 224, 224]}},
    # "mobilenet_v2": {"input0": [1, 32, 56, 56]},
    "dcgan": {1: {"input0": [1, 100]},
              8: {"input0": [8, 100]},
              16: {"input0": [16, 100]}},
    "yolov3": {1: {"input0": [1, 3, 416, 416]}},
    "mnist": {1: {"input0": [1, 1, 28, 28]}}
}


def create_relay_workload(expr):
    inputs = relay.analysis.free_vars(expr)
    expr_func = relay.Function(inputs, expr)
    mod, params = testing.create_workload(expr_func)

    return mod, params


def crop_expr_by_post_dfs_order(expr, post_dfs_order):
    return ExprCropper(post_dfs_order).crop(expr)


class ExprCropper:
    def __init__(self, target_post_dfs_order):
        self._memo = {}
        self._post_dfs_order = -1
        self._target_post_dfs_order = target_post_dfs_order
        self.target_expr = None

    def crop(self, expr):
        self._memo = {}
        self.target_expr = None
        self.visit_expr(expr)

        return self.target_expr

    # Visit Relay expressions in post-order
    def visit_expr(self, expr):
        if hash(expr) in self._memo:
            return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr)
            node_type = "Const"
        elif is_var_node(expr):
            self.visit_expr_var(expr)
            node_type = "Var"
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr)
            node_type = "TupleGetItem"
        elif is_call_node(expr):
            self.visit_expr_call(expr)
            node_type = expr.op
        elif is_function_node(expr):
            self.visit_expr_func(expr)
            node_type = "Function"
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr)
            node_type = "Tuple"
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

        self._post_dfs_order += 1
        if self._post_dfs_order == self._target_post_dfs_order:
            self.target_expr = expr

    def visit_expr_const(self, expr):
        pass

    def visit_expr_var(self, expr):
        pass

    def visit_expr_tuple(self, expr):
        for arg in expr.fields:
            self.visit_expr(arg)

    def visit_expr_tuplegetitem(self, expr):
        self.visit_expr(expr.tuple_value)

    def visit_expr_call(self, expr):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for arg in args:
            self.visit_expr(arg)

    def visit_expr_func(self, expr):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body)
