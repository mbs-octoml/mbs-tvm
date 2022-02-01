from tvm.relay.expr_functor import ExprVisitor
from collections import defaultdict
from tvm.ir import Op
import copy
from enum import IntEnum
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
from collage.utils import (
                        is_function_node,
                        get_group_id_from_backend_pattern_annotation,
                        get_backend_from_backend_pattern_annotation,
                        get_op_name_from_backend_pattern_annotation,
                    )  

EXT_COMPILERS = ["tensorrt", "dnnl"]

class BackendStateType(IntEnum):
    # This is for measurement
    FIRST_LEVEL_BEST_OP = 0
    TENSORRT_OP = 1

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

"""
This class translates optimized_match (Key: expression / Value: annotation (group_id + op_name))
into group_id_to_exprs_anno (Key: state id / Value: a list of pairs of expression and its annotation)

group_id_to_exprs_anno will be used for building state (search space) for evolutionary search
"""

class MatchToOpGroupTranslator(ExprVisitor):
    def __init__(self):
        super().__init__()

    def translate(self, expr, optimized_match):
        assert not is_function_node(expr)

        self.memo_map = {}
        self._optimized_match = optimized_match
        self.group_id_to_exprs_anno = defaultdict(list)
        self.visit(expr)

        return self.group_id_to_exprs_anno

    # Visit Relay expressions in post-order
    def visit(self, expr):
        if expr in self.memo_map:
            return
        else:
            # Op should be skipped
            if not isinstance(expr, Op):
                assert expr in self._optimized_match

                anno = self._optimized_match[expr]
                # Note that group id is passed as a string
                group_id = int(get_group_id_from_backend_pattern_annotation(anno))
                # backend_pattern_name = get_backend_pattern_name_from_backend_pattern_annotation(anno)

                # Group id is same as state id
                self.group_id_to_exprs_anno[group_id].append((expr, anno))

        super().visit(expr)


def is_invalid_ext_compiler_op_tensorrt(expr):
    is_not_valid = False
    # It's fine to allow TensorRT to take tuple or tuplegetitem
    # It could even allow TensorRT to merge more regions in comp graph
    # is_not_valid = is_tuple_node(expr) or is_tuplegetitem_node(expr)

    # Patterns that TensorRT can't afford
    transpose_pat = is_op("transpose")(wildcard())
    batch_matmul_pat = is_op("nn.batch_matmul")(wildcard(), wildcard())
    image_resize_pat = is_op("image.resize")(wildcard())

    variance_pat = is_op("variance")(wildcard(), wildcard())
    reshape_pat = is_op("reshape")(wildcard())
    divide_pat = is_op("divide")(wildcard(), wildcard())

    # is_not_valid |= transpose_pat.match(expr)
    # is_not_valid |= batch_matmul_pat.match(expr)
    is_not_valid |= image_resize_pat.match(expr)

    # For BERT full version
    # is_not_valid |= variance_pat.match(expr)
    # is_not_valid |= reshape_pat.match(expr)
    # is_not_valid |= divide_pat.match(expr)

    return is_not_valid

def is_invalid_ext_compiler_op_dnnl(expr):
    is_valid = False

    # Patterns that DNNL can afford
    is_valid |= str_to_pattern["CONV2D"].match(expr)
    is_valid |= str_to_pattern["CONV3D"].match(expr)
    is_valid |= str_to_pattern["BATCHNORM"].match(expr)
    is_valid |= str_to_pattern["DENSE"].match(expr)
    is_valid |= str_to_pattern["RELU"].match(expr)
    is_valid |= is_constant_or_var_node(expr)

    if str_to_pattern["RELU"].match(expr):
        shape = expr.checked_type.shape
        if len(shape)!=4:
            is_valid = False

    return not is_valid


"""
This back-translates state_id_to_exprs_anno into optimized_match for measurement;
It translates state_id_to_exprs_anno (Key: state id / Value: a list of pairs of expression and its annotation)
into optimized_match (Key: expression / Value: annotation (group_id + op_name))
"""

backend_to_invalid_op_checker = {
    "tensorrt": is_invalid_ext_compiler_op_tensorrt,
    "dnnl": is_invalid_ext_compiler_op_dnnl,
}

class OpStateToMatchTranslator():
    def __init__(self, optimized_match, group_id_to_exprs_anno, backend_name = "tensorrt"):
        self.optimized_match = optimized_match
        self.group_id_to_exprs_anno = group_id_to_exprs_anno

        self.graph_opt_backend_name = backend_name
        self.is_invalid_ext_compiler_op = backend_to_invalid_op_checker[self.graph_opt_backend_name]

        self.state_id_to_group_id = self.get_valid_op_state_by_filtering()

    def is_valid_op_state(self, expr_anno_pairs):
        assert len(expr_anno_pairs) > 0

        # To check if ops in each group have same backend
        expr, anno = expr_anno_pairs[0]
        first_backend = get_backend_from_backend_pattern_annotation(anno)
        is_valid_op_state = True

        for expr, anno in expr_anno_pairs:
            # To check if ops in each group have same backend
            backend_name = get_backend_from_backend_pattern_annotation(anno)
            assert backend_name == first_backend

            # If one of ops is tuple or tuple_get_item, then prevent it from being ext compiler ops
            if self.is_invalid_ext_compiler_op(expr):
                is_valid_op_state = False
                #if 'conv2d' in anno:
                #    print(f"{anno} is invalid")
                #    print(type(expr), anno)
                break

        # If this is External compiler op chosen from the first op optimizing pass,
        # Then we don't need to consider it on the second subgraph optimizing pass.
        if first_backend in EXT_COMPILERS:
            is_valid_op_state = False

        #print("-"*30)
        #if is_valid_op_state:
        #    print(f"{anno} : is valid? {is_valid_op_state}")

        return is_valid_op_state


    # Valid op states include ops that are not assigned to TensorRT
    # and that are not Tuple or TupleGetItem (These shouldn't be TensorRT ops)
    def get_valid_op_state_by_filtering(self):
        state_id_to_group_id = {}
        state_id = 0
        group_ids = self.group_id_to_exprs_anno.keys()
        for group_id in group_ids:
            expr_anno_pairs = self.group_id_to_exprs_anno[group_id]
            if self.is_valid_op_state(expr_anno_pairs):
                state_id_to_group_id[state_id] = group_id
                state_id += 1
        # print(state_id_to_group_id)
        # sys.exit(0)
        return state_id_to_group_id

    def gen_ext_compiler_op_annotation(self, anno):
        group_id = int(get_group_id_from_backend_pattern_annotation(anno))
        op_name = get_op_name_from_backend_pattern_annotation(anno)
        backend_name = self.graph_opt_backend_name

        return f"{group_id}-{backend_name}_{op_name}"

    # It only changes op to TensorRT op now
    def update_opt_match(self, state_id, new_opt_match):
        # print(new_opt_match)
        gid = self.state_id_to_group_id[state_id]
        for expr, anno in self.group_id_to_exprs_anno[gid]:
            assert expr in new_opt_match
            new_opt_match[expr] = self.gen_ext_compiler_op_annotation(anno)
            # print(f"anno / new anno: {anno} / {new_opt_match[expr]}")

    def translate(self, op_state):
        # Warning(@Soo): if you do deepcopy, it will create new instances copied from elements of list or dict
        new_opt_match = self.optimized_match.copy()

        # Translate individual into match
        # gid represents op group id (or op id)
        # backend_type represents 0 or 1
        # 0 - backend op selected from first level
        # 1 - tensorrt op
        for state_id, backend_type in enumerate(op_state):
            # print(backend_type)
            if backend_type == BackendStateType.TENSORRT_OP:
                # print("tensorrt")
                self.update_opt_match(state_id, new_opt_match)

        # sys.exit(0)

        return new_opt_match
#
# class OpState:
#     def __init__(self, state_id, exprs, annotation):
#         self._state_id =  state_id
