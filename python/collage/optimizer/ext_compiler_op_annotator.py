from tvm.relay.expr_functor import ExprMutator
#from collage.utils import *
import tvm.relay as relay
from collage.utils import printe

"""
This class annotates relay expression with compiler_begin and compiler_end
for each operator from target backend.
This will be merged by merge and partition pass following this.
"""
class ExtCompilerOpAnnotator(ExprMutator):
    def __init__(self, opt_match):
        super().__init__()
        self.opt_match = opt_match

    # read csv file and translate it into opt_match dictionary
    # (Key: expression / Value: annotation (group_id + op_name))
    def annotate(self, expr, target_str):
        self.target_str = target_str
        expr = self.visit(expr)
        return expr

    def is_target_op(self, expr):
        assert expr in self.opt_match

        is_target_op = False
        annotation = self.opt_match[expr]
        if get_backend_from_backend_pattern_annotation(annotation) == self.target_str:
            is_target_op = True

        return is_target_op

    # Visit Relay expressions in post-order
    def visit(self, expr):
        return super().visit(expr)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]

        # Wanring(@soo): Is it okay?
        if self.is_target_op(call):
            new_args = [relay.annotation.compiler_begin(new_arg, self.target_str) for new_arg in new_args]

        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if self.is_target_op(call):
            new_call = relay.annotation.compiler_end(new_call, self.target_str)

        return new_call

    # def visit_function(self, fn):
    #     new_params = [self.visit(x) for x in fn.params]
    #     new_body = self.visit(fn.body)
    #     return Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
    #
    # def visit_var(self, var):
    #     return var
    #
    # def visit_tuple(self, tup):
    #     # raise NotImplementedError
    #     return Tuple([self.visit(field) for field in tup.fields], tup.span)
    #
    # def visit_tuple_getitem(self, op):
    #     # raise NotImplementedError
    #     tuple_value = self.visit(op.tuple_value)
    #     if not tuple_value.same_as(op.tuple_value):
    #         return TupleGetItem(tuple_value, op.index)
    #     return op
    #
    # def visit_constant(self, const):
    #     return const