import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np

from collage.utils import (
                        is_var_node, 
                        is_constant_node, 
                        is_tuple_node, 
                        is_tuplegetitem_node,
                        get_op_pattern,
                        is_call_node,
                        get_args,
                        is_var,
                    )
from tvm.relay.dataflow_pattern import (
                        is_op, 
                        wildcard, 
                        is_tuple_get_item, 
                        is_tuple, is_constant, 
                        WildcardPattern,
                        CallPattern,
                        ConstantPattern,
                        VarPattern,
                    )
from .dp_table import (
                        FrontierQueue, 
                        MatchInfoExtractor, 
                        DPTable,
                    )
from .ordered_pattern_matcher import OrderedPatternMatcher
from collage.pattern_manager.default_patterns import relayop_to_varnames

from collage.analysis import visualize_backend_placement
import logging

# extract the subgraph of the expr that matches the pattern (only the top layers of the recursive relay expr).
# Since there might be multiple branches, we traverse each branch by "depth" steps, and rewrite the child nodes
# of the last node to free variables. However, when there are multiple branches, only the rewrite at the end of the
# longest branch will be useful


def extract_subgraph(expr, pattern):
  # use this to make sure the nodes (exprs) are consistent between different branches
  # Warning(@Soo): This is not necessary when we don't have diamond patterns
  # Still, this case happens in NasNet-A, e.g., addition of same avgpool2d results
  old_expr_to_new = dict()

  depth = pattern.get_depth()
  # print(f"depth: {depth}")

  relay_pattern = pattern.get_relay_pattern()
  def set_old_expr_to_new(expr, new_expr):
    old_expr_to_new[expr] = new_expr

  def get_expr(expr):
    if expr in old_expr_to_new:
      return old_expr_to_new[expr]
    return expr

  def helper(expr, depth, relay_pattern):
    assert relay_pattern.match(expr), f"(relay_pattern, expr) = ({relay_pattern}, {expr.op}, {expr.args[0].op})"
    # Warning(@Soo): To resolve NasNet-A corner case, e.g., addition of same avgpool2d results
    cur_checked_type = expr.checked_type
    expr = get_expr(expr)

    # assert (type(expr) is not tvm.ir.type.TupleType)
    if isinstance(relay_pattern, WildcardPattern):
      # The above issue with avgpool2d is resolved!
      # The problem is because checked_type is updated when generating new expr.
      # We resolved it by saving checked_type from old expression
      ret = relay.var("data", cur_checked_type)
      #if is_call_node(expr) or is_tuple_node(expr) or is_constant_node(expr):
      #  ret = relay.var("data", expr.checked_type)
      #else:
      #  ret = relay.var("data", expr.type_annotation)
      return ret
    elif isinstance(relay_pattern, ConstantPattern):
      return expr
    elif is_call_node(expr):
      # note that only call node has "op" attribute corresponding to a single backend operator
      op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span
      # if expr.op.name == 'reshape':
      #   print(f"New shape for reshape op : {expr.attrs.newshape}")
      new_args = []
      # at depth 1, turn call expr arguments into free variables with the same attributes and data shapes!
      if depth == 1:
        var_names = relayop_to_varnames[op.name]
        # # of arguments should match # of type arguments
        # Fix: This happens in BERT. We need to deal with it
        # It means that type inference hasn't been executed (type_args are not filled)
        # because inputs are variables, not relay op expr
        if len(expr.args) != len(expr.type_args):
          raise Exception("The type inference pass hasn't been executed.")
        else:
          # print(expr.op, var_names)
          for i in range(len(expr.args)):
            type_arg = expr.type_args[i]
            var_name = var_names[i]

            # Tuple should be treated separately
            if (type(type_arg) is tvm.ir.type.TupleType):
                input_data = expr.args[i]
                new_args.append(relay.Tuple([relay.var(var_name, d) for i, d in enumerate(type_arg.fields)] ))
          
            # Bias should be constant
            elif var_name == 'bias':
              input_data = expr.args[i].data
              new_args.append(relay.Constant(input_data))
            else:
              new_args.append(relay.var(var_name, type_arg))
      else:
        for c_idx, child in enumerate(expr.args):
          pat_child = relay_pattern.args[c_idx]
          new_args.append(helper(child, depth - 1, pat_child))

      args = new_args

      new_expr = tvm.relay.expr.Call(op, args, attrs, type_args, span)
      set_old_expr_to_new(expr, new_expr)
      return new_expr

    elif is_tuple_node(expr):
      new_args = []
      for c_idx, child in enumerate(expr.fields):
        pat_child = relay_pattern.fields[c_idx]
        new_args.append(helper(child, depth - 1, pat_child))
      new_expr = relay.Tuple(new_args)
      set_old_expr_to_new(expr, new_expr)

      return new_expr

    elif is_tuplegetitem_node(expr):
      tuple_value = helper(expr.tuple_value, depth, relay_pattern.tuple)
      new_expr = tvm.relay.expr.TupleGetItem(tuple_value, expr.index)
      set_old_expr_to_new(expr, new_expr)
      return new_expr

    elif is_var_node(expr):
      return expr

    elif is_constant_node(expr):
      return expr

    else:
      raise Exception(f"Expr type not implemented {type(expr)}")
  return helper(expr, depth, relay_pattern)

# given a pattern and a relay expr matching that pattern, return the cheapest backend operator
# satisfying the constraints and its cost. Return None if no backend operators satisfy constraints.
def get_optimal_backend_pattern(pattern_registry, expr, pattern, given_backends = None, build_target = "INVALID",
                          need_tvm_fallback_ops=False, fallback_backend_pats=None):
  assert type(given_backends) == list
  assert given_backends is not None

  backend_patterns = pattern_registry.get_backend_patterns(pattern)
  cheapest_bp, min_cost = None, float('inf')

  for bp in backend_patterns:
    # Check if its backend is configured or not.
    if bp.get_backend() not in given_backends:
      continue

    subgraph = extract_subgraph(expr, pattern)

    # Print the useful logs
    logging.info("-" * 45)
    # Warning(@Soo): there is a bug in printing repr of tuple in TVM.
    if is_tuple_node(subgraph):
      logging.info(f"Subgraph to measure (backend: {bp._backend.name}): {subgraph}")
    else:
      logging.info(f"Subgraph to measure (backend: {bp._backend.name}): {repr(subgraph)}")
    cost = bp.get_cost(subgraph, build_target)
    logging.info(f"Cost of subgraph : {cost:4f}")
    logging.info("-" * 45)

    assert cost != None
    if cost < min_cost:
      min_cost = cost
      cheapest_bp = repr(bp)

  # If no operator matched current patterns, fall back on TVM (no tuning) ops
  if cheapest_bp is None:
    assert need_tvm_fallback_ops

    if pattern in fallback_backend_pats:
      # For this pattern, no other backends than TVM can afford this pattern
      min_cost = 100000 # sys.maxsize
      # Even if it's an autotvm, it will lower to TVM (no-tuning) ops cuz we will build without AutoTVM logs
      # We name it as autotvm because current lowering pipeline does not differentiate between TVM and AutoTVM.
      cheapest_bp = f'autotvm-{pattern.get_name()}' # fallback op name
  else:
    # Exceptional cases to block
    # - 1) tensorrt_0-Op(add)[*, *] - If the add is the sole operator, then TensorRT has an issue to execute it
    if cheapest_bp == 'tensorrt_0-Op(add)[*, *]' and len(expr.checked_type.shape) != 4:
      cheapest_bp = f'autotvm-{pattern.get_name()}' # fallback op name

  if min_cost == float('inf') and not need_tvm_fallback_ops:
    raise Exception("No corresponding backend operators / or backend op errors out (e.g., CuDNN conv_bias_relu)")

  return cheapest_bp, min_cost


class CompGraphOptimizer:
    def __init__(self, pattern_registry, given_backends=None):
        self._pattern_registry = pattern_registry
        self._given_backends = given_backends
        self._ordered_pattern_matcher = OrderedPatternMatcher()
        # Attribute key to pass to N-to-1 lowering pass
        self._bop_attr_key = "backend-op"

        # With BackendList Attr, we do not have full op coverage without AutoTVM (e.g., if cuDNN is sole backend)
        # Thus, we need to use TVM fallback ops (without auto-tuninng)
        # Even if it is named as AutoTVM, we will build without tuning logs. So, effectively, it is TVM with no tuning.
        self._need_tvm_fallback_ops = False

        self.loc2match = None
        self._memo = None

        # @sunggg: Add driver cost
        # self.C = 0.01


    def optimize(self, comp_graph, build_target):
        # HACKY: Reset matched_expr
        comp_graph.reset()

        frontier_queue = FrontierQueue()
        frontier_queue.put(comp_graph.get_root())

        extractor = MatchInfoExtractor(comp_graph)
        dp_table = DPTable(self._pattern_registry, comp_graph)

        root_expr = comp_graph.get_root().get_relay_expr()
        dom_tree = relay.analysis.construct_dom_tree(root_expr, post_dom = False)
        post_dom_tree = relay.analysis.construct_dom_tree(root_expr, post_dom = True)
        self._ordered_pattern_matcher.add_dom_tree(dom_tree)

        # @sunggg: run all pattern generators
        all_exprs = []
        def _traverse_expr(node, node_list):
            if node in node_list:
                return
            node_list.append(node)        
        relay.analysis.post_order_visit(root_expr, lambda expr: _traverse_expr(expr, all_exprs))

        for expr in all_exprs:
            for backend in self._given_backends:
                if backend in self._pattern_registry.backends_with_pattern_generators:
                    generated_patterns = backend.pattern_generator.generate(post_dom_tree, expr)
                    for pattern in generated_patterns:
                        self._pattern_registry.add_backend_pattern(backend, pattern, None)
        
        logging.info(f"# Registered patterns")
        backend_pattern_map = dict()
        for pat in self._pattern_registry.all_backend_patterns:
          backend_name = pat.get_backend().name
          if backend_name not in backend_pattern_map:
            backend_pattern_map[backend_name] = set()
          backend_pattern_map[backend_name].add(pat.get_pattern())
        
        for backend, patterns in backend_pattern_map.items():
          logging.info(f"  >> backend name: {backend} ({len(patterns)})")
          for pat in patterns:
            logging.info(f"     - {pat}")
          logging.info(f"\n")
        
        # For backend ablation study where we are given a list of backends,
        # We need TVM (no-tuning) fall back operator patterns to have full op coverage
        # if AutoTVM is not inlcuded as a backend
        # Warning(@Soo): We need to discard TVM fallback operator fusion patterns that include ops supported by
        # backend in a list. It is currently dealt by the following codes.
        fallback_backend_pats = None
        while not frontier_queue.empty():
            # Facilitate the debugging process
            self._pattern_registry.save_to_log()
            f = frontier_queue.get()
            f_expr = f.get_relay_expr()

            logging.info("="*45)
            if is_call_node(f_expr):
                logging.info(f"(topo_order, pattern) : {f._topological_order}, {str(f_expr.op)}")
            else:
                logging.info(f"(topo_order, pattern) : {f._topological_order}, {type(f_expr)}, Non-call node")

            n_match_frontier = 0

            num_matches = 0
            for backend_pattern in self._pattern_registry.all_backend_patterns:
                pat = backend_pattern.get_pattern()
                backend = backend_pattern.get_backend()

            #for pat in self._pattern_registry.get_all_patterns():
                # print("Checking... ", pat)

                # ordered_pattern_matcher consider the order of arguments when matching
                # in contrast to basic Relay pattern matching.
                # If we don't use this, we need to modify extract_subgraph (for op measurement)
                if self._ordered_pattern_matcher.match(f_expr, pat.get_relay_pattern()):
                # if pat.get_relay_pattern().match(f_expr):
                    assert pat.get_depth() >= 1 # 0 depth doesn't make sense
                    logging.info(f"The following pattern is matched: {pat.get_relay_pattern()}")

                    # Get best backend op and its cost for matched nodes
                    best_backend_pattern_name, min_cost = get_optimal_backend_pattern(self._pattern_registry, f_expr,
                                                                           pat, self._given_backends, build_target,
                                                                           self._need_tvm_fallback_ops,
                                                                           fallback_backend_pats)

                    # Skip update if there is no backend op available for matched pattern
                    if best_backend_pattern_name is None:
                        continue

                    num_matches += 1
                    # Extract match information; refer to detailed explanation in the MatchInfoExtractor
                    matched_nodes, match_dic, new_frontiers = extractor.extract(f_expr, pat.get_relay_pattern(), best_backend_pattern_name)

                    dp_table.update(matched_nodes, match_dic, best_backend_pattern_name, min_cost, new_frontiers)

                    # Add new frontiers to the queue
                    prev_qsize = frontier_queue._frontiers.qsize()
                    frontier_queue.put(new_frontiers)
                    n_match_frontier += frontier_queue._frontiers.qsize() - prev_qsize
            
            if num_matches == 0:
              print(f_expr, type(f_expr))
              assert 0

        # Assign backend operator annotation (group_id + backend_pattern_name) to Relay expr (backend attribute)
        optimized_match = dp_table.assign_backend_pattern_to_expr()

        return optimized_match

    def match_pat_from_list(self, f_expr, backend_pats_ops, extractor, frontier_queue, group_id):
        # Match operators with target backend ops
        n_match_frontier = 0
        is_matched = False
        backend_annotation = "default"
        for pat, b_op in backend_pats_ops:
            # print("Checking... ", pat)

            if self._ordered_pattern_matcher.match(f_expr, pat.get_relay_pattern()):
                # if pat.get_relay_pattern().match(f_expr):
                assert pat.get_depth() >= 1  # 0 depth doesn't make sense
                # print("The following pattern is matched:", pat.get_relay_pattern())

                # Extract match information; refer to detailed explanation in the MatchInfoExtractor
                is_matched, b_op_name = True, repr(b_op)
                matched_nodes, match_dic, new_frontiers = extractor.extract(f_expr, pat.get_relay_pattern(), b_op_name)

                # Add new frontiers to the queue
                prev_qsize = frontier_queue._frontiers.qsize()
                frontier_queue.put(new_frontiers)
                n_match_frontier += frontier_queue._frontiers.qsize() - prev_qsize

                # Update backend in the Relay expression directly
                for expr, op_name in match_dic.items():
                    backend_annotation = create_backend_pattern_annotation(group_id, op_name)
                    # printe(f"Pair of type and annotation: {backend_annotation}")
                    # printe(repr(expr), backend_annotation)
                    relay.analysis.update_backend(expr, backend_annotation)

                # We match the longest possible backend ops, thus we stop here;
                # Note that patterns are sorted in the decreasing order of pattern depth
                break

        return is_matched, frontier_queue, backend_annotation

    def optimize_single_backend(self, comp_graph, single_backend):
        assert 0, "Disabled for demo"
        

class AssignBackendExprVisitor:
    def __init__(self):
        self._memo = {}

    def assign(self, expr, annotation):
        self._memo = {}
        self._annotation = annotation
        self.visit_expr(expr)

    # Visit Relay expressions in post-order
    def visit_expr(self, expr):

        if hash(expr) in self._memo:
           return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True
            relay.analysis.update_backend(expr, self._annotation)

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr)
        elif is_var_node(expr):
            self.visit_expr_var(expr)
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr)
        elif is_call_node(expr):
            self.visit_expr_call(expr)
        elif is_function_node(expr):
            self.visit_expr_func(expr)
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr)
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

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

