from tvm.relay.dataflow_pattern import *

VAR_CONST_STR = "Var/Const"
WILDCARD_STR = "*"

def find_depth(relay_pattern):
    if relay_pattern is None:
        return 0
    children = []

    if isinstance(relay_pattern, TuplePattern):
        children = relay_pattern.fields
    elif isinstance(relay_pattern, TupleGetItemPattern):
        children = [ relay_pattern.tuple ]
    elif isinstance(relay_pattern, CallPattern):
        children = relay_pattern.args
    elif isinstance(relay_pattern, WildcardPattern) or isinstance(relay_pattern, VarPattern) or isinstance(relay_pattern, ConstantPattern):
        return 0
    else:
        raise Exception(f"{type(relay_pattern)} is not handled yet.")

    depth = 0
    if children is not None:
        for child in children:
            depth = max(depth, find_depth(child))
    return depth+1


def get_name_and_children_relay_pattern(pattern, node_str):
    goDeeper = True

    if isinstance(pattern, TuplePattern):
        node_str += "Tuple"
        children = pattern.fields
    elif isinstance(pattern, TupleGetItemPattern):
        node_str += "TupleGetItem"
        children = [ pattern.tuple ]
    elif isinstance(pattern, WildcardPattern):
        node_str += WILDCARD_STR
        goDeeper = False
        children = []
    elif isinstance(pattern, VarPattern) or isinstance(pattern, ConstantPattern):
        node_str += VAR_CONST_STR
        goDeeper = False
        children = []
    elif isinstance(pattern, CallPattern):
        if isinstance(pattern.op, VarPattern) or isinstance(pattern.op, ConstantPattern):
            node_str += VAR_CONST_STR
            goDeeper = False
            children = []
        else:
            node_str += str(pattern.op)
            children = pattern.args
    else:
        raise Exception(f"{type(pattern)} is not handled yet.")

    return node_str, children, goDeeper


"""
This is used to extract names of Relay pattern;
Specifically, it is needed for single backend baseline execution to see if any part of the pattern can't be supported by 
the given single backend. If it is supported by the given single backend, we shouldn't consider it as a fallback backend op.
"""
def get_name_set_relay_pattern(pattern):
    node_str = ""
    node_str, children, goDeeper = get_name_and_children_relay_pattern(pattern, node_str)

    name_set = set()
    if node_str not in [VAR_CONST_STR, WILDCARD_STR]:
        name_set.add(node_str)

    if goDeeper:
        for i, child in enumerate(children):
            child_names = get_name_set_relay_pattern(child)
            name_set |= child_names

    return name_set


def name_relay_pattern(pattern, idMap = None, cnt = 0):
    if idMap is None:
        idMap = dict()

    if isinstance(pattern, WildcardPattern):
        node_str = ""
    else:
        if (pattern not in idMap):
            idMap[pattern] = cnt
            cnt += 1

        uid = idMap[pattern]
        node_str = f"{uid}-"

    node_str, children, goDeeper = get_name_and_children_relay_pattern(pattern, node_str)

    if goDeeper:
        name = node_str + "["
        for i, child in enumerate(children):
            if i > 0:
                name += ", "
            _name, _cnt = name_relay_pattern(child, idMap, cnt)
            name += _name
            cnt = _cnt

        name += "]"
    else:
        name = node_str

    return name, cnt


# currently the Pattern class does not add any additional attributes from TVM's dataflow patterns
class Pattern(object):
    def __init__(self, relay_pattern, name = None):
        self._relay_pattern = relay_pattern
        self._name = name_relay_pattern(relay_pattern)[0]
        self._depth = find_depth(relay_pattern)

    def __eq__(self, another):
        return isinstance(another, Pattern) and self._relay_pattern == another._relay_pattern

    def __hash__(self):
        return hash(self._name)

    def __repr__(self):
        return self._name

    def match(self, expr):
        return self._relay_pattern.match(expr)

    def get_name(self):
        return self._name

    def get_relay_pattern(self):
        return self._relay_pattern

    def get_depth(self):
        return self._depth


    """
    Get name of ops included in this pattern
    
    For sinlge backend baseline, we need this to see if the fallback pattern (e.g., AutoTVM) can be used to match;
    To be specific, if this fallback pattern does not include pattern from the target single backend, it can be used.
    e.g., For CuDNN baseline measurement, Conv + Relu (AutoTVM) can't be used because CuDNN supports Conv and RELU.
    However, we have to use Add from TVM because we can't support CuDNN Add in TVM (due to the PackedFunc issue).  
    """
    def get_op_name_set(self):
        return get_name_set_relay_pattern(self._relay_pattern)
