import tvm
from tvm import relay
from collage.utils import (
        is_constant_node, 
        is_var_node,
        is_tuplegetitem_node,
        is_tuple_node,
        is_call_node,
    )
from .optimizer_utils import is_data_var_node, is_data_node

class CGNode:
    def __init__(self, relay_expr, topological_order):
        self._relay_expr = relay_expr
        self._topological_order = topological_order
        self._hash_val = hash(relay_expr)
        self._parents = []
        self.matched_expr = {}

    def set_node_idx(self, idx):
        self.idx = idx

    def get_n_matched(self):
        return len(self.matched_expr.keys())

    def add_parent(self, parent_hash):
        self._parents.append(parent_hash)
        
    def get_n_parents(self):
        return len(self._parents)
    
    def get_parents(self):
        return self._parents
        
    def get_relay_expr(self):
        return self._relay_expr
    
    def __gt__(self, other):
        return self._topological_order > other._topological_order
    
    def __hash__(self):
        return self._hash_val

class ComputationGraph:
    def __init__(self, relay_expr):
        self._relay_expr = relay_expr

        # This is # of nodes except for constant and data
        self._memo = {}
        self._n_nodes = self._get_n_nodes(relay_expr)

        # This counts all relay nodes including constant and data
        self._memo = {}
        self.n_relay_nodes = self._get_n_nodes(relay_expr, is_relay_nodes=1)

        self._nodes = []
        self.expr2node = {}
        self._memo = {}
        self._expr2graph(relay_expr=relay_expr, topological_order=0, parent_expr=None)
        self._nodes.sort(key=lambda x: x._topological_order)

        # Set node idx for DP
        for idx, node in enumerate(self._nodes):
            node.set_node_idx(idx)

        assert self._n_nodes == len(self._nodes)
    
    def reset(self):
        for node in self._nodes:
            node.matched_expr = {}
        
    def get_relay_expr(self):
        return self._relay_expr
    
    def get_n_nodes(self):
        return self._n_nodes
        
    def _get_n_nodes(self, relay_expr, is_relay_nodes=0):
        self._memo[hash(relay_expr)] = True
        n_nodes = 1
        if is_constant_node(relay_expr) or (is_var_node(relay_expr) and not is_data_node(relay_expr)):
            n_nodes = is_relay_nodes
        elif is_data_var_node(relay_expr):
            n_nodes = 1
        elif is_tuplegetitem_node(relay_expr):
            next_expr = relay_expr.tuple_value
            if hash(next_expr) not in self._memo:
                n_nodes += self._get_n_nodes(next_expr, is_relay_nodes)
        elif is_tuple_node(relay_expr):
            for node_idx, node in enumerate(relay_expr.fields):
                if hash(node) not in self._memo:
                    # memorize this visit to prevent it from visiting twice
                    # +1 here means counting the current node
                    n_nodes += self._get_n_nodes(node, is_relay_nodes)
        elif is_call_node(relay_expr):
            for node_idx, node in enumerate(relay_expr.args):
                if hash(node) not in self._memo:
                    # memorize this visit to prevent it from visiting twice
                    # +1 here means counting the current node
                    n_nodes += self._get_n_nodes(node, is_relay_nodes)
        else:
            raise Exception(f"Unexpected Relay expr type {type(relay_expr)}")

        return n_nodes
   
    def get_root(self):
        return self._nodes[0]

    def _add_node(self, relay_expr, topological_order, parent_expr):
        node_idx = len(self._nodes)
        self._nodes.append(CGNode(relay_expr, topological_order))
        self.expr2node[hash(relay_expr)] = self._nodes[-1]

    def _expr2graph(self, relay_expr, topological_order, parent_expr):
        if is_constant_node(relay_expr) or (is_var_node(relay_expr) and not is_data_node(relay_expr)):
            return
        else:
            if hash(relay_expr) not in self._memo:
                self._add_node(relay_expr, topological_order, parent_expr)
                self.expr2node[hash(relay_expr)].add_parent(parent_expr)
                self._memo[hash(relay_expr)] = True
            else:
                self.expr2node[hash(relay_expr)].add_parent(parent_expr)
                if self.expr2node[hash(relay_expr)]._topological_order < topological_order:
                    self.expr2node[hash(relay_expr)]._topological_order = topological_order
                else:
                    return

            if is_data_var_node(relay_expr):
                return
            elif is_tuple_node(relay_expr):
                for node_idx, node in enumerate(relay_expr.fields):
                    self._expr2graph(node, topological_order + 1, relay_expr)
            elif is_tuplegetitem_node(relay_expr):
                # If it is tuple, you should use tuple_value instead of args
                next_expr = relay_expr.tuple_value
                self._expr2graph(next_expr, topological_order+1, relay_expr)
            elif is_call_node(relay_expr):
                for node_idx, node in enumerate(relay_expr.args):
                    self._expr2graph(node, topological_order + 1, relay_expr)
            else:
                raise Exception("Unexpected Relay expr type")