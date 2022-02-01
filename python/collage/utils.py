from tvm import relay
from tvm.relay.dataflow_pattern import *
from collections import namedtuple
import numpy as np
import tvm

from .pattern_manager.pattern_language import Pattern
import logging
import datetime
import sys
import os
import logging


# With more argumentsc
def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        # sys.stdout.flush()
        # sys.stderr.flush()

    def flush(self):
        pass

# Add new logging level
MYINFO = 35
logging.addLevelName(MYINFO, 'MYINFO')
def myinfo(self, message, *args, **kws):
    self.log(MYINFO, message, *args, **kws)
logging.Logger.myinfo = myinfo

def setup_logging(log_dir, task_name, net_name, hw_name, logging_level=logging.WARNING, batch_size=1):
    date_now = datetime.datetime.now()
    this_code_path = os.path.dirname(os.path.abspath(__file__))
    date_now = date_now.strftime("%m-%d-%H:%M")
    file_path = f"./logs/{log_dir}/{task_name}_{hw_name}_{net_name}_bs{batch_size}_{date_now}.log"

    logging.basicConfig(filename=file_path, level=logging_level,
                        format='%(asctime)s:[%(levelname)s] %(message)s')

    log = logging.getLogger('logger')


    sys.stdout = StreamToLogger(log, MYINFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

# an example of a diamond pattern that occurs in resnet-18
def get_diamond():
  relu = is_op("nn.relu")(wildcard())
  conv2d1_1 = is_op("nn.conv2d")(relu, wildcard())
  batch_norm1 = is_tuple_get_item(is_op("nn.batch_norm")(conv2d1_1, wildcard(), wildcard(), wildcard(), wildcard()), 0)
  relu1 = is_op("nn.relu")(batch_norm1)
  conv2d1_2 = is_op("nn.conv2d")(relu1, wildcard())
  conv2d2 = is_op("nn.conv2d")(relu, wildcard())
  diamond = is_op("add")(conv2d1_2, conv2d2)
  return Pattern(diamond)

def is_data_tensor(ndarr):
  assert isinstance(ndarr, tvm.runtime.NDArray)
  return np.sum(ndarr.asnumpy()) == 0


def convert_shape_imm_to_tuple(shape_imm):
  return tuple(map(lambda x: x.value, shape_imm))

def extract_input_shape(node, input_shape_arr, memo):
  # Prevent it from vising same node more than once
  if node in memo:
    return
  else:
    memo[node] = True

  # Update expensive operator stats
  if is_var_node(node):
    input_shape_arr.append(convert_shape_imm_to_tuple(node.type_annotation.shape))
  elif is_constant_node(node):
    input_shape_arr.append(convert_shape_imm_to_tuple(node.data.shape))

# Note that this expr is always subgraph used for operator measurement
def get_input_shape(expr):
  input_shape_arr, memo = [], {}
  relay.analysis.post_order_visit(expr, lambda node: extract_input_shape(node, input_shape_arr, memo))
  return tuple(input_shape_arr)

# Deprecated(@Soo): This one doesn't include the parameter shape
# However, we need parameter shape too to decide whether configuration of op 1 and op 2 are same or not.
# e.g., conv doesn't have attribute of output channel size / matmul doesn't have output shape

# def get_data_shape(expr):
#   inputs = relay.analysis.free_vars(expr)
#   # if is_call_node(expr):
#   #   print(f"Input for expr ({expr.op}) {[inputs[0].type_annotation]}, {expr.attrs.axis}")
#
#   # for add, shape of lhs and rhs should be identical. for all other backend ops, we take shape of "data" input arg
#   # inputs[0] corresponds to Var(name_hint='data')
#   # We consider two different types for that: TupleTypeNode, TensorTypeNode
#   if type(inputs[0].type_annotation) == relay.TensorType:
#     data_shape_imm = inputs[0].type_annotation.shape
#     data_shape = list(map(lambda x: x.value, data_shape_imm))
#   elif type(inputs[0].type_annotation) == relay.TupleType:
#     data_shape = []
#     for tup_item in inputs[0].type_annotation.fields:
#       data_shape.append(tuple((map(lambda x: x.value, tup_item.shape))))
#     data_shape = tuple(data_shape)
#     print("data shape", data_shape)
#   else:
#     raise Exception(f"Unsupported Var type ({type(inputs[0].type_annotation)})")
#
#   return data_shape

# Deprecated (@Soo)
# def get_data(expr):
#   data_shape = get_data_shape(expr)
#   if type(data_shape) == list:
#     print(data_shape)
#     data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
#   elif type(data_shape) == tuple:
#     data = []
#     for shape in data_shape:
#       data.append(np.random.uniform(-1, 1, size=shape).astype("float32"))
#       print(shape)
#     data = tuple(data)
#   else:
#     raise Exception(f"Unsupported data shape type {type(data_shape)}")
#
#   return data


"""
Helper functions for parsing backend op name annotation
e.g., 0-tensorrt_conv2d+relu -> 1, 0-tvm_conv2d_+relu -> 0
"""
def create_backend_pattern_annotation(group_id, annotation):
  return f"{group_id}-{annotation}"

def get_group_id_from_backend_pattern_annotation(annotation):
  return annotation.split("-")[0]

def get_backend_pattern_name_from_backend_pattern_annotation(annotation):
  backend_pattern_name_pos = annotation.find("-")+1
  return annotation[backend_pattern_name_pos:]

def get_backend_from_backend_pattern_annotation(annotation):
  return get_backend_pattern_name_from_backend_pattern_annotation(annotation).split("_")[0]

def get_op_name_from_backend_pattern_annotation(annotation):
  op_name_pos = annotation.find("_") + 1
  return annotation[op_name_pos:]

def get_group_id_and_backend_pattern_name(annotation):
  return get_group_id_from_backend_pattern_annotation(annotation), get_backend_pattern_name_from_backend_pattern_annotation(annotation)



def is_function_node(expr):
  return type(expr) == tvm.relay.Function

def is_constant_node(expr):
  return type(expr) == tvm.relay.Constant

def is_call_node(expr):
  return type(expr) == tvm.relay.expr.Call

def is_tuple_node(expr):
  return type(expr) == tvm.relay.expr.Tuple

def is_tuplegetitem_node(expr):
  return type(expr) == tvm.relay.expr.TupleGetItem

def is_call_or_tuplegetitem_node(expr):
  # If not, it means that we need to add codes to deal with other nodes
  assert is_call_node(expr) or is_tuplegetitem_node(expr) or is_var_node(expr) or is_constant_node(expr) or is_tuple_node(expr)
  return is_call_node(expr) or is_tuplegetitem_node(expr)

def is_var_node(expr):
  return type(expr) == tvm.relay.expr.Var

def is_constant_or_var_node(expr):
  return is_constant_node(expr) or is_var_node(expr)

def no_constraints_func(config):
  return True

def get_function_body(expr):
  assert is_function_node(expr)
  return expr.body

# given a tvm.ir.Attrs node, return list of attribute values
def get_attr_vals(expr):
  assert is_call_node(expr)
  attrs = expr.attrs
  op_name = expr.op.name

  if attrs == None or "keys" not in dir(attrs):
    return (op_name, "")

  # print(f"{expr.op}'s Attrs : {dir(attrs)}")
  keys = attrs.keys()
  values = []
  for key in keys:
    value = attrs.get_int(key)
    v_type = type(value)

    if v_type == type(None):
      pass
    elif v_type == tvm.ir.container.Array:
      value = tuple(value)
    elif v_type == int or v_type == str or v_type == float:
      pass
    elif v_type == tvm.tir.expr.IntImm:
      value = int(value)
    elif v_type == tvm.runtime.container.String:
      value = tuple(value)
    else:
      print(key, value, v_type)
      raise Exception(f"Unexpected tvm data type ({v_type}) for attributes")

    values.append(value)

  return (op_name, tuple(zip(keys, values)))

# extract the node attributes of a relay expr. Use a list of tvm node attributes to represent each path (branch) in the expr.
# Then use an immutable set of these lists to represent all node attributes of the expr.
# Note that expr is extracted subgraph that matches backend op
def extract_attrs(expr):
  res = set()

  def helper(expr, attrs):
    if is_call_node(expr):
      attr_vals = get_attr_vals(expr)
      # print(attr_vals)
      attrs += attr_vals
      children = list(filter(is_call_or_tuplegetitem_node, expr.args))
      if len(children) == 0:
        res.add(attrs)

      for child in children:
        helper(child, attrs)

    elif is_tuplegetitem_node(expr):
      helper(expr.tuple_value, attrs)

    elif is_tuple_node(expr):
      children = list(filter(is_call_or_tuplegetitem_node, expr.fields))
      if len(children) == 0:
        res.add(attrs)

      for child in children:
        helper(child, attrs)

    elif is_var_node(expr):
      raise Exception("Should not reach var node")

    else:
      raise Exception("Expr type not implemented")

  helper(expr, ())
  return frozenset(res)

class OptLevel():
  def __init__(self, opt_level):
    self.opt_level = opt_level

  def set(self, opt_level):
    self.opt_level = opt_level

  def get(self):
    return self.opt_level

# extract the node attributes of a relay expr. Use a list of tvm node attributes to represent each path (branch) in the expr.
# Then use an immutable set of these lists to represent all node attributes of the expr.
# def extract_attrs(expr):
#   if is_call_node(expr):
#     attrs += get_attr_vals(expr)
#     children = list(filter(is_call_or_tuplegetitem_node, expr.args))
#     if len(children) == 0:
#       return attrs

#     for child in children:
#       extract_attrs(child)

#   elif is_tuplegetitem_node(expr):
#     extract_attrs(expr.tuple_value, attrs)

#   elif is_var_node(expr):
#     raise Exception("Should not reach var node")

#   else:
#     raise Exception("Expr type not implemented")

#   helper(expr, ())
#   return res


def serialize_subgraph(subgraph):
  def _traverse(node, visited, node_list):
    if node in visited:
        return
    if isinstance(node, tvm.ir.op.Op):
        return

    visited.add(node)
    if is_constant_node(node) or is_var_node(node):
        node_list.append("$")
    else:
        node_list.append(node.op.name)

  visited = set()
  node_list = []
  relay.analysis.post_order_visit(subgraph, lambda node: _traverse(node, visited, node_list))
  return "/".join(node_list)


def get_op_pattern(expr):
    if is_tuple_node(expr):
        return 7 # kTuple: hardcoded for now
    elif is_call_node(expr):
        return expr.op.get_attr("TOpPattern")
    elif isinstance(expr, tvm.ir.op.Op):
        return expr.get_attr('TOpPattern')
    else:
        return 8 # kOpaque


def get_args(node):
    if is_tuple_node(node):
        return node.fields
    #elif is_tuplegetitem_node(node):
    #    return is_tuple_get_item, 2
    elif is_call_node(node):
        return node.args
    elif is_constant_node(node) or is_var_node(node):
        return []
    else:
        raise Exception(f"Unsupported type ({type(node)})")

