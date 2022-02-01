# Immutable and efficient implementation for an array of bits
from bitarray import frozenbitarray
from .comp_graph import (
            CGNode,
        )
from collage.utils import (
            is_constant_or_var_node,
            is_var_node, 
            is_constant_node, 
            is_tuple_node, 
            is_tuplegetitem_node,
            get_op_pattern,
            is_call_node,
            get_args,
            is_var,
        )
from collage.optimizer.optimizer_utils import log_matched_ops_by_method
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
from collections import defaultdict
import logging
from collage.utils import (
            create_backend_pattern_annotation,
            get_backend_from_backend_pattern_annotation
        )
from tvm import relay
from collage.interface import CollageContext


try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

"""
FrontierQueue class
- It maintains frontiers to match patterns using PriorityQueue
- Current main job is to prevent duplicate frontiers from being added to PriorityQueue
"""

class FrontierQueue:
    def __init__(self):
        self._frontiers = Q.PriorityQueue()
        self._memo = {}

    def put_item(self, item):
        # print(type(item))
        assert type(item) == CGNode

        if item not in self._memo:
            self._memo[item] = True
            self._frontiers.put(item)

    def put(self, item):
        if type(item) == list:
            for it in item:
                self.put_item(it)
        else:
            self.put_item(item)

    def empty(self):
        return self._frontiers.empty()

    def get(self):
        return self._frontiers.get()

"""
MatchInfoExtractor aims to collect following two information
1) Matched nodes
2) New frontiers after match
3) Match dictionary where a key is Relay expr and a value is backend op annotation

Example
- Expr: Add(Conv(Data, Weight), Data2) Pattern: Add(*, *)
- 1) Add(..)
- 2) Conv(Data, Weight) # Data2 doesn't count because it's not an op
- 3) {Add(...) : '0-tvm-add'}
"""
class MatchInfoExtractor:
    def __init__(self, comp_graph):
        self._comp_graph = comp_graph

    def extract(self, expr, pattern, op_name):
        self._memo = {}
        self.matched_nodes = []
        self.new_frontiers = []

        # Collect match information (Key: Relay Expr, Value: backend op name)
        self.op_name = op_name
        self.match_dic = {}

        self.visit_expr(expr, pattern)

        return self.matched_nodes, self.match_dic, self.new_frontiers

    # Visit Relay expressions in post-order
    def visit_expr(self, expr, pattern):
        # Warning(@Soo): What if we have a pattern that matches the same node twice? e.g., a leaf of diamond pattern
        # If the generated pattern is invalid, this could be an issue. But, let's assume we only have valid patterns.
        if expr in self._memo:
            return
        else:
            self._memo[expr] = True

        if is_constant_or_var_node(expr):
            self.match_dic[expr] = self.op_name

            # Warning(@Soo): Comment this because data node can be matched multiple times,
            # so we should exclude it from matched_nodes. It will still be included in match_dic
            # # Corner case: Var for input data should be considered as a node
            # if is_data_var_node(expr):
            #     node = self._comp_graph.expr2node[hash(expr)]
            #     self.matched_nodes.append(node)

        else:
            node = self._comp_graph.expr2node[hash(expr)]

            # Add current expr to new frontier if wildcard
            if isinstance(pattern, WildcardPattern):
                # Note that Data Node (Var) won't be included in new frontiers
                self.new_frontiers.append(node)
                return
            else:
                self.match_dic[expr] = self.op_name
                self.matched_nodes.append(node)

            if is_tuplegetitem_node(expr):
                self.visit_expr_tuplegetitem(expr, pattern)
            elif is_call_node(expr):
                self.visit_expr_call(expr, pattern)
            elif is_tuple_node(expr):
                self.visit_expr_tuple(expr, pattern)
            else:
                raise Exception(f"Unexpected expression type, {type(expr)}")

    def visit_expr_tuple(self, expr, pattern):
        for a_idx, arg in enumerate(expr.fields):
            self.visit_expr(arg, pattern.fields[a_idx])

    def visit_expr_tuplegetitem(self, expr, pattern):
        self.visit_expr(expr.tuple_value, pattern.tuple_value)

    def visit_expr_call(self, expr, pattern):
        op, args, attrs, type_args, span = expr.op, expr.args, expr.attrs, expr.type_args, expr.span

        for a_idx, arg in enumerate(args):
            self.visit_expr(arg, pattern.args[a_idx])


"""
DPTableCell class
- It keeps necessary information for last optimal match for a DP table cell
 > e.g., opt_cost, opt_pat, opt_match
"""
class DPTableCell:
    def __init__(self, best_b_op_cost, best_b_op_name, prev_cell, match_dic,
                 key, post_order_bits, min_post_dfs_order):#, frontiers):
        # Last one matched pattern when this cell is created
        # We have to backtrace to get all patterns included in the optimal match using prev_cell below.
        self.best_b_op_name = best_b_op_name

        # match_dic only includes match with best_b_op_name
        self.match_dic = match_dic

        # Pointer to the previous DPTableCell including optimal match before current match
        # It is used to get all patterns included in the optimal match
        self.prev_cell = prev_cell

        # @sunggg: This is a driver cost to get correct cost estimate when operators are not fused
        # Guideline(@Soo): it should be hardware-dependent ideally.
        # For now, we use 0.01 for everything else than bert-full (0.5)
        # Bert-full has the issue of picking up inefficeint TensorRT ops for lightweighted kernel
        # e.g., add+relu, multiply, etc; it is more likely the issue of op measurement variance though
        self.C = 0.01 #0.5 #0.1 #1 #0.01

        # Optimal cost with selected nodes
        self.opt_cost = self.get_cost(best_b_op_cost, prev_cell)

        # Frontiers of matches so far
        # self.frontiers = frontiers

        # match_post_dfs_order k to which all nodes up to k post_dfs_order are matched
        # min_post_dfs_order means the min of match_post_dfs_order
        # given that we started matching for min_post_dfs_order and nodes up to min should be all matched
        self.key = key
        self._post_order_bits = post_order_bits
        self.match_post_dfs_order = self.get_match_post_dfs_order(min_post_dfs_order)

    def __repr__(self):
        return f"{self.best_b_op_name}, {str(self.match_dic)}"

    def is_matched_up_to_post_dfs_order(self, post_dfs_order, key):
        p_bits = self._post_order_bits[post_dfs_order]
        return p_bits & self.key == p_bits

    def get_match_post_dfs_order(self, min_order):
        match_order = min_order
        while match_order in self._post_order_bits:
            if not self.is_matched_up_to_post_dfs_order(match_order, self.key):
                match_order -= 1
                break
            # else:
            #     raise Exception("It means that match_order can be more than min_order, which doesn't make sense")
            match_order += 1

        # This assertion doesn't hold: for example, conv2d+Relu can be 3 vs. -1
        # assert match_order == min_order or match_order == min_order+1, f"match_order {match_order} vs. min_order {min_order}"

        return match_order

    def get_cost(self, op_cost, prev_cell):
        # prev_opt_cost means optimal cost for all patterns matched before the current matched pattern.
        prev_opt_cost = prev_cell.opt_cost if prev_cell is not None else 0
        return prev_opt_cost + op_cost + self.C

    # If DPTableCell already exists for a given a DPTable key,
    # then we need to update an existing one if new match is better than the current best
    def update(self, op_cost, op_name, prev_cell, match_dic):
        total_cost = self.get_cost(op_cost, prev_cell)
        if total_cost < self.opt_cost:
            self.opt_cost = total_cost
            self.best_b_op_name = op_name
            self.prev_cell = prev_cell
            self.match_dic = match_dic

"""
DPTable class
- It maintains the DP table (dictionary) where key is a set of matched nodes in bits (e.g., 0110...)
and value is a tuple of min cost and last matched_pattern.
- It can also generate a string of all matched patterns and a result dictionary where
key is a relay Expression (pointer) and value is a matched backend operator annotation.
"""
class DPTable:
    def __init__(self, pattern_registry, comp_graph):
        self._pattern_registry = pattern_registry
        self._comp_graph = comp_graph

        self._n_nodes = comp_graph.get_n_nodes() - 1
        logging.info(f"# of nodes in comp graph: {self._n_nodes}")
        root_node = comp_graph.get_root()
        default_key_str = ''.join(self.gen_default_node_key_list())
        self._zero_key = frozenbitarray(default_key_str)

        # This is a hash table where a key is a matched node and value is a key for DPTableCells to be updated.
        # Note that these DPTableCells do not include matched nodes, but include the parent of the root of matched nodes.
        self._node_to_key = defaultdict(set)
        # -1 is because we set topological order from 0 and there is a dummy match at the beginning
        min_post_dfs_order = -1
        tmp_key = self.gen_key_for_node_to_key_dic(min_post_dfs_order)
        self._node_to_key[tmp_key].add(self._zero_key)

        # Pruning: We aim to check if DPTableCell includes all the nodes up to k-th post_dfs_order
        # using pre-generated bits that mean all the nodes up to k-th post_dfs_order are matched
        # _post_order_bits has a key of post_dfs_order (k) and value of bits
        self._post_order_bits = self.gen_post_order_bits()
        self._dp_table = {self._zero_key: DPTableCell(0, None, None, {},
                                                      self._zero_key, self._post_order_bits, min_post_dfs_order)}  # , {root_node})}
        # print(self._post_order_bits)

    def gen_key_for_node_to_key_dic(self, match_post_dfs_order):
        return match_post_dfs_order
        # This doesn't include all the frontiers for newly created match
        # For example, it misses frontiers from previous match before merging with matched_frontiers
        # return (node.idx, match_post_dfs_order)

    def gen_post_order_bits(self):
        key_list = self.gen_default_node_key_list()
        post_dfs_order = 0
        # Dummpy post_order_bits
        post_order_bits = {-1: self.gen_node_key_from_key_list(key_list)}

        # Note that _topological_order corresponds to post_dfs_order
        for node_idx in range(len(key_list)):
            if self._comp_graph._nodes[node_idx]._topological_order > post_dfs_order:
                node_key = self.gen_node_key_from_key_list(key_list)
                post_order_bits[post_dfs_order] = node_key
                post_dfs_order = self._comp_graph._nodes[node_idx]._topological_order

            key_list[node_idx] = '1'

        assert post_dfs_order not in post_order_bits
        node_key = self.gen_node_key_from_key_list(key_list)
        post_order_bits[post_dfs_order] = node_key

        return post_order_bits

    # def __repr__(self):
    #     return self._dp_table

    # Generate a key of DPTable, which is an array of bits, the n-th of which indicates whether n-th node is matched or not (0 or 1)
    # e.g., 1000 means that only first node is matched
    # Input: a list of matched nodes
    # Output: a key of DPTable
    def gen_node_key(self, matched_nodes):
        key_list = self.gen_default_node_key_list()
        for node in matched_nodes:
            key_list[node.idx] = '1'
        return self.gen_node_key_from_key_list(key_list)

    def gen_node_key_from_key_list(self, key_list):
        return frozenbitarray(''.join(key_list))

    # Generate a default key of DPTable (a series of 0)
    def gen_default_node_key_list(self):
        return ['0' for _ in range(self._n_nodes)]

    def get_parent_nodes(self, node):
        assert type(node) == CGNode
        parent_nodes = []
        for parent_expr in node.get_parents():
            # This means node is a root (top) node in Relay Expr, which is final result node in computation graph
            if parent_expr is None:
                continue

            parent = self._comp_graph.expr2node[hash(parent_expr)]
            parent_nodes.append(parent)

        return parent_nodes

    def get_root_matched_nodes(self, matched_nodes):
        return matched_nodes[0]

    def are_parents_included(self, node, key):
        parent_nodes = self.get_parent_nodes(node)
        flag = False

        if len(parent_nodes) == 0:
            flag = True
        else:
            parents_key = self.gen_node_key(parent_nodes)
            flag = (parents_key & key) == parents_key
            # check if at least one parent is included
            # flag = (parents_key & key) != self._zero_key
            # printe(f"(key, parent_key, flag) = ({key}, {parents_key}, {flag})")

        return flag

    # def gen_new_frontiers(self, prev_cell, matched_nodes, matched_frontiers):
    #     # Get new frontiers for new DPTableCell by merging previous frontiers with matched_frontiers
    #     new_frontiers = set.union(prev_cell.frontiers, matched_frontiers)
    #
    #     # 1) Remove previous frontiers if matched by matched_nodes
    #     for prev_frontier in prev_cell.frontiers:
    #         if prev_frontier in matched_nodes:
    #             new_frontiers.remove(prev_frontier)
    #
    #     # 2) Add matched frontiers if not matched by matched_nodes
    #     for m_frontier in matched_frontiers:
    #         # This can potentially be improved
    #         m_frontier_key = self.gen_node_key([m_frontier_key])
    #         if m_frontier in


    # Generate candidate cells to update based on one of following strategies
    # 1) generate candidates that do not include current matched nodes
    # 2) generate candidates that do not include current matched nodes & include at least one of parents for root matched nodes
    # 3) generate candidates that do not include current matched nodes & include parents of root matched nodes
    #    & include post-dominators of all matched nodes (on computation graph, not Relay Expr)
    # Note that 2) holds under the assumption that we only have patterns with a single root
    # We figured out that 1) is too slow, so stick to 2).
    # 3) requires additional implementation for building post-dominator tree
    def gen_candidate_cells(self, root_matched_node, matched_nodes, min_order):#, frontiers):
        # If matching happens from k post_dfs_order, then it means that all nodes up to post_dfs_order of k
        # should already be matched.
        candidates = []
        cur_match_key = self.gen_node_key(matched_nodes)

        # printe(f"[gen_candidate_cells] node_to_key: {self._node_to_key}")
        tmp_key = self.gen_key_for_node_to_key_dic(min_order)
        # printe(f"[gen_candidate_cells] cur_match_key, min_dfs_order: {cur_match_key}, {min_order}")
        candidate_keys = self._node_to_key[tmp_key]

        for key in candidate_keys:
            # printe(f"[gen_candidate_cells] candidate key: {key}")
            cell = self._dp_table[key]

            # and self.are_parents_included(root_matched_node, key):
            if (cur_match_key & key) == self._zero_key and cell.match_post_dfs_order >= min_order:
                assert cell.match_post_dfs_order == min_order

                new_key = cur_match_key | key
                # printe(f"[gen_candidate_cells] Selected, new key: {new_key}")
                candidates.append((new_key, cell))

                # Deal with frontiers
                # new_frontiers = self.gen_new_frontiers(cell, matched_nodes, frontiers)


        return candidates

    def update(self, matched_nodes, match_dic, best_backend_pattern_name, min_cost, frontiers):
        # Generate candidate DPTableCells that need to be updated with new match
        root_matched_node = self.get_root_matched_nodes(matched_nodes)
        match_post_dfs_order = root_matched_node._topological_order
        min_order = match_post_dfs_order - 1

        candidate_cells = self.gen_candidate_cells(root_matched_node, matched_nodes, min_order)

        # Update DPTableCells with new match
        for (new_key, prev_cell) in candidate_cells:
            if new_key not in self._dp_table:
                self._dp_table[new_key] = DPTableCell(min_cost, best_backend_pattern_name, prev_cell, match_dic,
                                                      new_key, self._post_order_bits, min_order)
            else:
                self._dp_table[new_key].update(min_cost, best_backend_pattern_name, prev_cell, match_dic)

            cell = self._dp_table[new_key]
            cell_order = cell.match_post_dfs_order

            if cell_order >= min_order:
                # This assertion doesn't hold: for example, conv2d+Relu can be 3 vs. -1
                # assert cell_order in [min_order, min_order + 1], f"cell_order {cell_order} vs. min_order {min_order}"
                # printe(f"[update] Added cell_order, key: {cell_order}, {new_key}")
                tmp_key = self.gen_key_for_node_to_key_dic(cell_order)
                self._node_to_key[tmp_key].add(new_key)

                # Pruning condition 1: All parents of root (of matched nodes) should be included in candidates
                # if self.are_parents_included(root_matched_node, new_key):
                #     self._node_to_key[node].add(new_key)

    def assign_backend_pattern_to_expr(self):
        all_matched_key = frozenbitarray('1'*self._n_nodes)
        #print(self._node_to_key)
        # This is a cell representing the first match;
        opt_match_cell = self._dp_table[all_matched_key]

        # Note that last prev_cell is always None
        logging.info("="*50)
        logging.info("Matched operators (in post-dfs-order, from the root of comp graph to the last node)")
        group_id, backend_annotation = 0, None
        optimized_match = {}

        # Match annotation
        matched_b_op_name = []
        # Note that we have one dummy cell, and that's why it's opt_match_cell.prev_cell instead of opt_match_cell
        while opt_match_cell.prev_cell is not None:
            # Warning(@Soo): It might be important which backend to assign to data node
            # For now, it can be assigned to any parallel ops randomly.
            # Let's keep that in mind
            for expr, op_name in opt_match_cell.match_dic.items():
                backend_annotation = create_backend_pattern_annotation(group_id, op_name)
                relay.analysis.update_backend(expr, backend_annotation)
                optimized_match[expr] = backend_annotation

            logging.info(f"{backend_annotation}")
            matched_b_op_name.append(backend_annotation)

            opt_match_cell = opt_match_cell.prev_cell
            group_id += 1

        logging.info("=" * 50)
        #log_matched_ops_by_method(CollageContext.op_level_placement, matched_b_op_name)

        return optimized_match
