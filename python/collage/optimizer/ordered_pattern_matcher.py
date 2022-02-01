from tvm import relay
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

class OrderedPatternMatcher:
    def __init__(self):
        self.matched_exprs = set()
        # @sunggg
        # TODO: This is not the best form since these two are like global variable within the object.
        # It's better to use local variable but leave them for now for the sake of time.
        self.doms_of_matched_exprs = set()
        self.dom_tree = None

    # TODO: test if there is a cycle.
    #   e.g.,   Conv
    #          /   \
    #        Relu  Sigmoid
    #          \   /
    #           Add
    # Although there is a pattern for Conv-Relu-Add, we should not match this one because of the cycle dependency.

    def add_dom_tree(self, _dom_tree):
        self.dom_tree = _dom_tree


    def get_doms_of_matched_exprs(self):
        def traverse(expr):
            if expr in self.dom_tree:
                dom = self.dom_tree[expr]
                self.doms_of_matched_exprs.add(dom)
                traverse(dom)

        for expr in self.matched_exprs:
            traverse(expr)

    def match(self, expr, pattern):
        assert(self.dom_tree is not None)
        # Initialize
        self.matched_exprs = set()
        self.doms_of_matched_exprs = set()
        is_matched = self.visit_expr(expr, pattern)

        if is_matched:
            assert(expr in self.matched_exprs)
            if len(self.matched_exprs) == 1:
                return True
            else:
                self.get_doms_of_matched_exprs()
                return not relay.analysis.has_cycle(expr, list(self.matched_exprs), list(self.doms_of_matched_exprs))
        else:
            return False


    # Visit Relay expressions in post-order
    def visit_expr(self, expr, pattern):
        is_matched = True

        if not pattern.match(expr):
            return False

        if isinstance(pattern, WildcardPattern):
            return True

        # If the code reaches here, it meas that expr matches and the pattern is not Wildcard.
        # Thus, we should add this to matched_exprs if it's not constant or var node
        # Note that we only save operator nodes as matched exprs
        if not is_constant_node(expr) and not is_var_node(expr):
             self.matched_exprs.add(expr)

        # We assume that child class at least have methods for these
        if is_constant_node(expr) or is_var_node(expr):
            pass
        elif is_tuplegetitem_node(expr):
            is_matched &= self.visit_expr_tuplegetitem(expr, pattern)
        elif is_call_node(expr):
            is_matched &= self.visit_expr_call(expr, pattern)
        elif is_tuple_node(expr):
            is_matched &= self.visit_expr_tuple(expr, pattern)
        else:
            raise Exception(f"Unexpected expression type, {type(expr)}")

        return is_matched

    # TODO: Why do we need separate function? Recursion across multiple functions does not seem desirable.
    def visit_expr_tuple(self, expr, pattern):
        is_matched = True
        for a_idx, arg in enumerate(expr.fields):
            is_matched &= self.visit_expr(arg, pattern.fields[a_idx])

        return is_matched

    def visit_expr_tuplegetitem(self, expr, pattern):
        is_matched = True
        is_matched &= self.visit_expr(expr.tuple_value, pattern.tuple_value)
        return is_matched

    def visit_expr_call(self, expr, pattern):
        is_matched = True
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for a_idx, arg in enumerate(args):
            is_matched &= self.visit_expr(arg, pattern.args[a_idx])

        return is_matched
