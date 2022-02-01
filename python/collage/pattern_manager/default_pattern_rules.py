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
                        TuplePattern,
                        TupleGetItemPattern,
                    )
from .base_pattern_rule import BasePatternRule, BasePatternGenerator
from collage.pattern_manager.pattern_language import Pattern
import tvm
import logging

# @ Sung: Check all nodes between src and sink by using fcheck (checks fusion conditions)
# Start from sink
# Gotta be dfs
def check_path(src, node, fcheck, path = [], paths = []):
    path.append(node)
    if src == node:
        assert(len(path))
        paths.append(path.copy())

    elif is_var_node(node) or is_constant_node(node):
        pass
    elif fcheck(node, node==src):
        children = []
        if is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        for child in children:
            check_path(src, child, fcheck, path, paths)

    out = path.pop()
    assert(node == out)
    


def generate_relay_pattern_node(node):
    # @sunggg: hacky solution to deal with tuple
    # Unlike is_op(), is_tuple() expects to accept operands at its initialization
    if is_tuple_node(node):
        return "tuple", len(node.fields)
    elif is_tuplegetitem_node(node):
        logging.info(node, dir(node))
        assert 0
        return is_tuple_get_item, 2
    elif is_call_node(node):
        return is_op(node.op.name), len(node.args)
    elif is_constant_node(node):
        return wildcard(), 0
        #return is_constant(), 0
    elif is_var_node(node):
        return wildcard(), 0
        #return is_var(), 0
    elif isinstance(node, tvm.ir.op.Op):
        return is_op(node.name), node.num_inputs
    else:
        raise Exception(f"Unsupported type ({type(node)})")


# NOTE: Seems like relay pattern matching considers pointer of pattern node.
#       Also, relay pattern string can be misleading
# for example,
#         is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
#         path1 = is_op('nn.relu')(is_conv2d)
#         path2 = is_op('nn.leaky_relu')(is_conv2d)
#         diamond = is_op('add')(path1, path2)
#         --> CallPatternNode(Op(add), [CallPatternNode(Op(nn.relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])]), CallPatternNode(Op(nn.leaky_relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])])])
#
#         This diamond pattern does not match with the following expr
#            inp1 = relay.var('input1')
#            inp2 = relay.var('input2')
#            weight1 = relay.var('weight1')
#            weight2 = relay.var('weight2')
#            conv2d1 = relay.op.nn.conv2d(inp1, weight1)
#            conv2d2 = relay.op.nn.conv2d(inp2, weight2)
#            relu = relay.op.nn.relu(conv2d1)
#            leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
#            out = relu + leaky_relu


def build_pattern_with_map(src, node, nodeToPatternMap):
    if node in nodeToPatternMap:
        rpattern = nodeToPatternMap[node][0]

        children = []
        if is_var_node(node) or is_constant_node(node):
            pass
        elif is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        if node == src:
            return rpattern

        operands = [ build_pattern_with_map(src, child, nodeToPatternMap) for child in children ]
        return rpattern(*operands)
    else:
        return wildcard()


def generate_relay_pattern(src, sink, paths = None, cur_pattern_type = None, nodeToPatternMap = dict()):
    if paths is None:
        # Handle single node
        assert(not (is_constant_node(src) and is_var_node(src)))
        assert(cur_pattern_type is None)
        rpattern, num_operands = generate_relay_pattern_node(sink)

        if num_operands == 0:
            assert(rpattern != "tuple")
            return rpattern, get_op_pattern(sink), 1

        operands = [wildcard() for __ in range(num_operands)]
        # @sunggg: hacky solution to deal with tuple
        if rpattern == "tuple":
            return is_tuple(operands), get_op_pattern(sink), 1
        else: 
            return rpattern(*operands), get_op_pattern(sink), 1

    else:
        # Handle multiple nodes
        # Create pattern node for all nodes in paths
        cnt = 0
        nodeToPatternMap = dict()
        for path in paths:
            for node in path:
                if node not in nodeToPatternMap:
                    nodeToPatternMap[node] = generate_relay_pattern_node(node)
                    cnt += 1

                # Create pattern node for const/var
                #for child in get_args(node):
                #    if is_constant_node(child) or is_var_node(child):
                #        if child not in nodeToPatternMap:
                #            nodeToPatternMap[child] = wildcard() #generate_relay_pattern_node(child)

        assert src in nodeToPatternMap, f"{src.op.name}"
        pnode, num_operands = nodeToPatternMap[src]

        if num_operands == 0:
            assert(pnode != "tuple")
            nodeToPatternMap[src] = (pnode, 0)
            rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)
            return rpattern, cur_pattern_type, cnt

        operands = [wildcard() for __ in range(num_operands)]

        # @sunggg: hacky solution to deal with tuple
        if pnode == "tuple":
            nodeToPatternMap[src] = (is_tuple(operands), 0) # it's zero cause we already handled.    
        else:
            nodeToPatternMap[src] = (pnode(*operands), 0) # it's zero cause we already handled.
        rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)

        return rpattern, cur_pattern_type, cnt


# @sunggg: default pattern generator. 
# To simulate recent fusion engines, it generates patterns with dom tree.
class DefaultPatternGenerator(BasePatternGenerator):
    def generate(self, post_dom_tree, expr):
        generated_patterns = list()
        if is_constant_node(expr) or is_var_node(expr):
            return generated_patterns # returns empty node

        # Check anchor node
        if is_tuple_node(expr) or self.pattern_rule.op_rule(expr):
            anchor_pattern, anchor_type, num_ops = generate_relay_pattern(expr, expr)
            
            # Verify if it is legitimate
            if self.pattern_rule.verify(anchor_pattern):
                generated_patterns.append(Pattern(anchor_pattern))
           
        
            def simulate_fusion(src, sink, cur_type, num_ops, nodeToPatternMap = dict()):
                assert(src is not None)
                if is_tuple_node(sink) and (sink in post_dom_tree):
                    simulate_fusion(src, post_dom_tree[sink], cur_type, num_ops, nodeToPatternMap)
                
                ops_to_fuse =  self.pattern_rule.fusion_rule(
                        src = src, 
                        sink = sink, 
                        cur_type = cur_type, 
                        num_ops = num_ops,
                    )
                if len(ops_to_fuse):
                    fusion_pattern, cur_type, num_ops = generate_relay_pattern(src, sink, ops_to_fuse, cur_type, nodeToPatternMap)
                
                    # Append identified pattern
                    if self.pattern_rule.verify(fusion_pattern):
                        generated_patterns.append(Pattern(fusion_pattern))

                    # Go deeper
                    if sink in post_dom_tree:
                        simulate_fusion(src=src, sink=post_dom_tree[sink], cur_type=cur_type, num_ops=num_ops,  nodeToPatternMap = nodeToPatternMap)
                
            
            # Run fusion simulation
            if expr in post_dom_tree:
                simulate_fusion(src=expr, sink=post_dom_tree[expr], cur_type=anchor_type, num_ops=num_ops)
                
        return generated_patterns
        

# This pattern rule should be singleton
class TVM_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256

    __instance = None
    @staticmethod
    def destroy():
        TVM_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TVM_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TVM_PatternRule.__instance = self

    @staticmethod
    def op_rule(expr):
        return get_op_pattern(expr) != TVM_PatternRule.optype2enum["kOpaque"]
    
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        enum2optype = TVM_PatternRule.enum2optype
        optype2enum = TVM_PatternRule.optype2enum
        MAX_NUM_OPS  = TVM_PatternRule.MAX_NUM_OPS

        def _check_path(src, sink, fcheck):
            def helper(src, node, fcheck, path = [], paths = []):
                path.append(node)

                if src == node:
                    assert(len(path))    
                    paths.append(path.copy())
                elif is_var_node(node) or is_constant_node(node):
                    pass
                elif fcheck(node, node==src):
                    children = []
                    if is_tuple_node(node):
                        children = node.fields
                    elif is_tuplegetitem_node(node):
                        children = [ node.tuple ]
                    elif is_call_node(node):
                        children = node.args
                    else:
                        raise Exception(f"Unsupported type ({type(node)})")
                
                    for child in children:
                        helper(src, child, fcheck, path, paths)
            
                out = path.pop()
                assert(node == out)
            
            path, paths = [], []
            helper(src, sink, fcheck, path, paths)
            
            return paths

        if num_ops > MAX_NUM_OPS:
            return list()

        sink_type = get_op_pattern(sink)

        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return list()

        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return list()
             
tvm_pattern_rule = TVM_PatternRule()
tvm_pattern_generator = DefaultPatternGenerator(tvm_pattern_rule)



class TRT_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256
    ops_to_exclude = ["image.resize"] # Not supported in TensorRT
    __instance = None
    
    @staticmethod
    def destroy():
        TRT_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TRT_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TRT_PatternRule.__instance = self

    @staticmethod
    def op_rule(expr):
        optype2enum = TRT_PatternRule.optype2enum
        ops_to_exclude = TRT_PatternRule.ops_to_exclude

        if is_call_node(expr):
            return (expr.op.name not in ops_to_exclude)

        if isinstance(expr, tvm.ir.op.Op):
            return (expr.name in ops_to_exclude)

        return (get_op_pattern(expr) != optype2enum["kOpaque"]) 
    

    # Based on TensorRT documentation
    # It seems almost same with TVM's fusion 
    # TODO: Shuffle-shuffle, shuffle-reduce patterns
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        # Borrow type definitions from TVM to ease implemenation overhead
        # Users can define their own if they want
        enum2optype = TVM_PatternRule.enum2optype
        optype2enum = TVM_PatternRule.optype2enum
        MAX_NUM_OPS  = 256

        def _check_path(src, sink, fcheck):
            def helper(src, node, fcheck, path = [], paths = []):
                path.append(node)

                if src == node:
                    assert(len(path))    
                    paths.append(path.copy())
                elif is_var_node(node) or is_constant_node(node):
                    pass
                elif fcheck(node, node==src):
                    children = []
                    if is_tuple_node(node):
                        children = node.fields
                    elif is_tuplegetitem_node(node):
                        children = [ node.tuple ]
                    elif is_call_node(node):
                        children = node.args
                    else:
                        raise Exception(f"Unsupported type ({type(node)})")
                
                    for child in children:
                        helper(src, child, fcheck, path, paths)
            
                out = path.pop()
                assert(node == out)
            
            path, paths = [], []
            helper(src, sink, fcheck, path, paths)
            
            return paths

        if num_ops > MAX_NUM_OPS:
            return list()

        sink_type = get_op_pattern(sink)

        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return list()

        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return list()

    @staticmethod
    def verify(pattern):
        q = [ pattern ]
        while len(q):
            cur = q.pop()
            if isinstance(cur, WildcardPattern):
                pass
            elif isinstance(cur, CallPattern):
                if ( 
                      isinstance(cur.op, ConstantPattern) 
                      or isinstance(cur.op, VarPattern)
                      or isinstance(cur.op, WildcardPattern)
                ):
                    pass
                else:
                    op_name = cur.op.expr.name

                    if op_name in TRT_PatternRule.ops_to_exclude:
                        return False
                    q.extend(cur.args)
            elif isinstance(cur, TuplePattern):
                q.extend(cur.fields)
            elif isinstance(cur, TupleGetItemPattern):
                q.append(cur.tuple_value)
            else:
                raise Exception(f"Unexpected expression type, {type(cur)}")

        return True

trt_pattern_rule = TRT_PatternRule() 
trt_pattern_generator = DefaultPatternGenerator(trt_pattern_rule)
