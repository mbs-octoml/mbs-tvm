import tvm
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np

from .optimizer_utils import get_pattern_len, get_next_expr_after_match
from ..pattern_manager.utils import *
from ..pattern_manager.backend_pattern import get_optimal_backend_pattern
from tvm.relay import ExprFunctor
from ..pattern_manager.target import *

try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

INVALID_ANNOTATION = "INVALID"

"""
ExtCompilerGroup is a group for external compiler operators, e.g., TensorRT, OpenVINO.
One group includes multiple external compiler operators from same external compilers. 
"""
class ExtCompilerGroup:
    def __init__(self, annotation):
        group_id = get_group_id_from_backend_pattern_annotation(annotation)
        backend_name = get_backend_from_backend_pattern_annotation(annotation)

        self.id = group_id
        self.exprs = []
        self.backend_name = backend_name
        self.op_name = ""

        # To prevent duplicate operator name
        self._memo = {}

    def get_annotation(self):
        return create_backend_pattern_annotation(self.id, self.backend_name + self.op_name)

    def add_op(self, expr, annotation):
        # If this is added before, we don't need to add op_name to group_op_name
        if annotation not in self._memo:
            self._memo[annotation] = True

            # Update backend_pattern_name and exprs
            op_name = get_op_name_from_backend_pattern_annotation(annotation)
            # "/" divides it into partitions before merging
            self.op_name += f"_{op_name}"

        self.exprs.append(expr)

"""
ExtCompilerGroupSet is a set of groups including external compiler operators, e.g., TensorRT, OpenVINO.
One set includes all groups for a given network. 
This class supports merging groups to one group 
"""
class ExtCompilerGroupMerger:
    def __init__(self):
        # To prevent duplicate group id
        self._group_id_to_group = {}

    def merge_two_groups(self, root_gid, gid):
        # print(root_gid, gid, self._group_id_to_group)
        assert self._group_id_to_group[root_gid].backend_name == self._group_id_to_group[gid].backend_name
        self._group_id_to_group[root_gid].exprs += self._group_id_to_group[gid].exprs
        self._group_id_to_group[root_gid].op_name += self._group_id_to_group[gid].op_name
        del self._group_id_to_group[gid]

    def merge(self, optimized_match, gid_to_root_gid):
        gid_arr = []
        # Create a group for each original group (before merge)
        for expr, anno in optimized_match.items():
            gid = get_group_id_from_backend_pattern_annotation(anno)
            if gid not in self._group_id_to_group:
                self._group_id_to_group[gid] = ExtCompilerGroup(anno)
                gid_arr.append(gid)
            self._group_id_to_group[gid].add_op(expr, anno)
        gid_arr.sort()

        # Merge groups into a root group
        for gid in gid_arr:
            root_gid = gid_to_root_gid[gid]
            if gid != root_gid:
                self.merge_two_groups(root_gid, gid)

        # Translate group to match
        merged_match = {}
        for gid, group in self._group_id_to_group.items():
            for expr in group.exprs:
                merged_match[expr] = group.get_annotation()

        return merged_match

"""
ExtCompilerOpMerger is for external compiler operators, e.g., TensorRT, OpenVINO.
ExtCompilerOpMerger groups contiguous ops from same external compilers
"""
class ExtCompilerOpMerger:
    def __init__(self, optimized_match):
        self._memo = {}

        self._optimized_match = optimized_match
        # gid = group id
        # root gid means a group id that ops will be merged into
        # We only records a root group id as a value to merge every group into a root group
        self._gid_to_root_gid = self.create_gid_to_root_gid(optimized_match)


    def create_gid_to_root_gid(self, optimized_match):
        gid_to_root_gid = {}
        for _, anno in optimized_match.items():
            gid = get_group_id_from_backend_pattern_annotation(anno)
            gid_to_root_gid[gid] = gid
        return gid_to_root_gid

    def merge(self, expr):
        self._memo = {}
        # prev_cur_backend_pattern_annotation includes group id for prev and cur nodes
        # e.g., 0-tensorrt_relu
        self.visit_expr(expr, (INVALID_ANNOTATION, INVALID_ANNOTATION))

        # gid_to_root_gid should be ready at this point
        # Goal: create a new optimized match based on merge results
        merged_match = ExtCompilerGroupMerger().merge(self._optimized_match, self._gid_to_root_gid)
        return merged_match

    def merge_two_groups(self, prev_op_anno, cur_op_anno):
        # Update root gid for current gid
        prev_group_id = get_group_id_from_backend_pattern_annotation(prev_op_anno)
        cur_group_id = get_group_id_from_backend_pattern_annotation(cur_op_anno)
        self._gid_to_root_gid[cur_group_id] = self._gid_to_root_gid[prev_group_id]

    # def create_group(self, cur_expr, cur_op_anno):
    #     cur_group_id = get_group_id_from_backend_pattern_annotation(cur_op_anno)
    #     if cur_group_id not in self._group_id_to_group:
    #         self._group_id_to_group[cur_group_id] = ExtCompilerGroup(cur_op_anno)
    #
    #     self._group_id_to_group[cur_group_id].add_op(cur_expr, cur_op_anno)

    def is_same_ext_compiler(self, prev_op_anno, cur_op_anno):
        prev_backend_name = get_backend_from_backend_pattern_annotation(prev_op_anno)
        cur_backend_name = get_backend_from_backend_pattern_annotation(cur_op_anno)
        return prev_backend_name in EXTERNAL_COMPILERS and prev_backend_name == cur_backend_name

    # Visit Relay expressions in post-order
    def visit_expr(self, expr, prev_cur_backend_pattern_annotation):
        prev_cur_backend_pattern_annotation = (prev_cur_backend_pattern_annotation[1], self._optimized_match[expr])
        prev_op_anno, cur_op_anno = prev_cur_backend_pattern_annotation

        if hash(expr) in self._memo:
            return
        else:
            # memorize this visit to prevent it from visiting twice
            self._memo[hash(expr)] = True

            # If cur expr is matched to the same external compiler op with previous expr,
            # then merge two ops in terms of optimized_match.
            if self.is_same_ext_compiler(prev_op_anno, cur_op_anno):
                self.merge_two_groups(prev_op_anno, cur_op_anno)

        # We assume that child class at least have methods for these
        if is_constant_node(expr):
            self.visit_expr_const(expr, prev_cur_backend_pattern_annotation)
            node_type = "Const"
        elif is_var_node(expr):
            self.visit_expr_var(expr, prev_cur_backend_pattern_annotation)
            node_type = "Var"
        elif is_tuplegetitem_node(expr):
            self.visit_expr_tuplegetitem(expr, prev_cur_backend_pattern_annotation)
            node_type = "TupleGetItem"
        elif is_call_node(expr):
            self.visit_expr_call(expr, prev_cur_backend_pattern_annotation)
            node_type = expr.op
        elif is_function_node(expr):
            self.visit_expr_func(expr, prev_cur_backend_pattern_annotation)
            node_type = "Function"
        elif is_tuple_node(expr):
            self.visit_expr_tuple(expr, prev_cur_backend_pattern_annotation)
            node_type = "Tuple"
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")


    def visit_expr_const(self, expr, prev_cur_backend_pattern_annotation):
        pass

    def visit_expr_var(self, expr, prev_cur_backend_pattern_annotation):
        pass

    def visit_expr_tuple(self, expr, prev_cur_backend_pattern_annotation):
        for arg in expr.fields:
            self.visit_expr(arg, prev_cur_backend_pattern_annotation)

    def visit_expr_tuplegetitem(self, expr, prev_cur_backend_pattern_annotation):
        self.visit_expr(expr.tuple_value, prev_cur_backend_pattern_annotation)

    def visit_expr_call(self, expr, prev_cur_backend_pattern_annotation):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for arg in args:
            self.visit_expr(arg, prev_cur_backend_pattern_annotation)

    def visit_expr_func(self, expr, prev_cur_backend_pattern_annotation):
        params, body, ret_type, type_params = expr.params, expr.body, expr.ret_type, expr.type_params
        self.visit_expr(body, prev_cur_backend_pattern_annotation)