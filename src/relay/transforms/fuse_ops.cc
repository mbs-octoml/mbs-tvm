/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file src/relay/transforms/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

/*
  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  However, at the point of conv2d we do not necessarily know that all the future paths
  will merge at the elemwise add. The fusion algorithm applies post-dominator analysis.

  The immediate post-dominator of a node defined by the closest node where all the future path goes
  into. In the above case, the elemwise add is the post-dominator of conv2d. The general algorithm
  is as follows:

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.

  Note that, because we run analysis on a DAG, we use a single pass post-dominator
  tree construction algorithm via LCA, which is simpler than the full version that handles cycles.

  The fusion algorithm traverses from each node and checks if it can be fused to its
  immediate post dominator. It has to check the following things:

  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  - Note that these intermediate node can already be fused with another nodes, the algorithm
      will still run correctly.
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/
using support::LinkedList;
using support::LinkNode;

constexpr uint32_t kMaxFusedOps = 256;

static const Op& stop_fusion_op = Op::Get("annotation.stop_fusion");

TVM_REGISTER_PASS_CONFIG_OPTION("relay.FuseOps.max_depth", Integer);

int NEW_BACKEND_GROUP_ID_FUSE_PASS = 30000000;
template <typename T>
void UpdateBackendWithNewGroup(tvm::relay::Expr op) {
  std::string new_backend = std::to_string(NEW_BACKEND_GROUP_ID_FUSE_PASS++) + "-autotvm";
  op.set_backend(new_backend);
}

// PATCH(@Soo): New data type for group id and backend op name
const std::string kInvalidBackendOp = "INVALID_BACKEND_OP";
const std::string kInvalidGroupIdOpNamePair = "9999999-INVALID_BACKEND_OP";

struct GroupIdOpNamePair {
  /*! \brief The group id for operators that will be fused */
  int group_id;
  /*! \brief The backend operator name */
  std::string backend_op_name;

  // Example pair_str: "0-tvm-default_batchflatten"
  // First number before '-' is the group id
  GroupIdOpNamePair(const std::string pair_str) {
    std::string delimiter = "-";
    int delim_pos = pair_str.find(delimiter);
    std::string group_id_str = pair_str.substr(0, delim_pos);

    // Initialization
    //        std::cerr << "pair_str: " << pair_str << std::endl;
    //        std::cerr << "group_id_str: " << group_id_str << std::endl;
    group_id = std::stoi(group_id_str);
    backend_op_name = pair_str.substr(delim_pos + 1);
    //        debug_print();
  }

  // Dummy constructor
  GroupIdOpNamePair() : GroupIdOpNamePair(kInvalidGroupIdOpNamePair) {}

  void debug_print() {
    // std::cerr << "Pair: " << group_id << "," << backend_op_name << std::endl;
  }
};

/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragment and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class IndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
    /*! \brief backend library to use. */
    std::string backend;
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < post_dfs_order.size(); ++i) {
      Node* node = post_dfs_order[i];
      os << "node[" << i << "], " << GetRef<ObjectRef>(node->ref) << " outputs=[";
      for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
        os << link->value.node->index << ", ";
      }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static IndexedForwardGraph Create(support::Arena* arena, const Expr& body);

 private:
  class Creator;
};

// Creator of post dominator tree of the dataflow
class IndexedForwardGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  IndexedForwardGraph Prepare(const Expr& body) {
    this->Update(body, nullptr, kOpaque);
    this->VisitExpr(body);
    return std::move(graph_);
  }

 private:
  /*! \brief allocator of all the internal node object */
  support::Arena* arena_;
  // The output.
  IndexedForwardGraph graph_;
  // attribute equal comparator
  StructuralEqual attr_equal_;
  // Update the message stored at the node.
  void Update(const Expr& node, IndexedForwardGraph::Node* parent, OpPatternKind pattern) {
    const tvm::Object* key = node.get();
    IndexedForwardGraph::Node* current;
    auto it = graph_.node_map.find(key);
    if (it != graph_.node_map.end()) {
      current = it->second;
    } else {
      current = arena_->make<IndexedForwardGraph::Node>();
      graph_.node_map[key] = current;
    }
    if (parent != nullptr) {
      auto* link = arena_->make<LinkNode<IndexedForwardGraph::Edge> >();
      link->value.node = parent;
      link->value.pattern = pattern;
      current->outputs.Push(link);
    } else {
      current->extern_ref = true;
    }
  }

  void AddNode(const tvm::Object* key) {
    auto it = graph_.node_map.find(key);
    ICHECK(it != graph_.node_map.end()) << "Cannot find node " << GetRef<ObjectRef>(key);
    IndexedForwardGraph::Node* node = it->second;
    ICHECK(node->ref == nullptr);
    node->ref = key;
    node->index = graph_.post_dfs_order.size();
    graph_.post_dfs_order.push_back(node);
  }

  // Post order tree
  void VisitExpr_(const FunctionNode* op) final {
    // Skip the function that should be handled by external codegen.
    if (op->GetAttr<String>(attr::kCompiler).defined()) return;

    for (auto param : op->params) {
      this->Update(param, nullptr, kOpaque);
    }
    this->Update(op->body, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ConstantNode* op) final {
    this->AddNode(op);
    Node* node = graph_.node_map.at(op);

    // Add backend to node
    node->backend = op->backend();

    DataType dtype = DataType(op->data->dtype);
    // This rule must be consistent with code generator.
    bool is_simple_const =
        (dtype == DataType::Int(32) || dtype == DataType::Int(64) || dtype == DataType::Float(32) ||
         dtype == DataType::Float(64) || dtype == DataType::Bool());
    if (op->is_scalar() && is_simple_const) {
      node->pattern = kElemWise;
    } else {
      // for now, mark non-scalar constant
      // as opaque, we will not choose to fuse it.
      node->pattern = kOpaque;
    }
  }

  void VisitExpr_(const CallNode* call) final {
    ICHECK(graph_.node_map.count(call));
    Node* node = graph_.node_map.at(call);

    // Add backend to node
    node->backend = call->backend();

    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    // Now we set the pattern of this call.
    //
    // If we see a call mentioning an operator we should mark it with its
    // annotated pattern.
    //
    // If the pattern is not annotated we will default to opaque.
    //
    // Finally if the operator position is not a call node we will
    // need to call Update, as it may be an arbitrary expression.
    OpPatternKind op_pattern = kOpaque;
    if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto op = GetRef<Op>(opnode);
      if (IsDynamic(call->checked_type()) && IsDataDependent(call)) {
        // output of a shape func can't be fed to a data-dependent shape func
        op_pattern = kOpaque;
      } else {
        op_pattern = static_cast<OpPatternKind>(fpattern[op]);
      }
    } else {
      this->Update(call->op, node, kOpaque);
    }

    node->pattern = op_pattern;
    this->Update(call->op, nullptr, kOpaque);
    const auto* rtype = call->checked_type().as<TensorTypeNode>();
    // pass the analysis back to all the children it references.
    for (size_t i = 0; i < call->args.size(); ++i) {
      const auto* arg_type = call->args[i]->checked_type().as<TensorTypeNode>();
      // specifically check if result type is the same as arguments type
      OpPatternKind edge_pattern = op_pattern;
      if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
          attr_equal_(rtype->shape, arg_type->shape)) {
        edge_pattern = kElemWise;
      }
      this->Update(call->args[i], node, edge_pattern);
    }
    ExprVisitor::VisitExpr_(call);
    this->AddNode(call);
  }

  void VisitExpr_(const TupleNode* op) final {
    ICHECK(graph_.node_map.count(op));
    Node* tuple_node = graph_.node_map.at(op);
    tuple_node->pattern = kTuple;
    for (const Expr& field : op->fields) {
      if (field->checked_type().as<TensorTypeNode>()) {
        this->Update(field, tuple_node, kInjective);
      } else {
        this->Update(field, nullptr, kOpaque);
      }
    }
    ExprVisitor::VisitExpr_(op);

    // Add backend to node
    tuple_node->backend = op->backend();

    this->AddNode(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    auto tuple_type = op->tuple->checked_type().as<TupleTypeNode>();
    ICHECK(tuple_type);
    // When TVM lowers a fused function, it expects all arguments to be a Tensor or
    // a tuple containing only Tensors. But this tuple may contain a reference or
    // another tuple. To avoid modifying codegen logic, we do not allow fusing through this node
    // if the tuple contains such non Tensor fields. However, all fields will be recursively
    // visited via call to ExprVisitor::VisitExpr_(op) below and corresponding visitor methods.
    bool has_non_tensor = false;
    for (auto ty : tuple_type->fields) {
      if (!ty.as<TensorTypeNode>()) {
        has_non_tensor = true;
        break;
      }
    }
    if (has_non_tensor) {
      this->Update(op->tuple, nullptr, kOpaque);
    } else {
      ICHECK(graph_.node_map.count(op));
      Node* node = graph_.node_map.at(op);
      node->pattern = kInjective;
      this->Update(op->tuple, node, kInjective);
    }
    ExprVisitor::VisitExpr_(op);

    // Add backend to node
    graph_.node_map.at(op)->backend = op->backend();

    this->AddNode(op);
  }

  void VisitExpr_(const VarNode* op) final {
    graph_.node_map.at(op)->backend = op->backend();
    this->AddNode(op);
  }

  void VisitExpr_(const LetNode* op) final {
    // do not fuse through let.
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      this->Update(op->var, nullptr, kOpaque);
      this->Update(op->value, nullptr, kOpaque);
      this->Update(op->body, nullptr, kOpaque);
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
      this->AddNode(op);
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }

  void VisitExpr_(const IfNode* op) final {
    // do not fuse through if.
    this->Update(op->cond, nullptr, kOpaque);
    this->Update(op->true_branch, nullptr, kOpaque);
    this->Update(op->false_branch, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefCreateNode* op) final {
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefReadNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const RefWriteNode* op) final {
    this->Update(op->ref, nullptr, kOpaque);
    this->Update(op->value, nullptr, kOpaque);
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }

  void VisitExpr_(const MatchNode* op) final {
    this->Update(op->data, nullptr, kOpaque);
    for (const Clause& c : op->clauses) {
      this->Update(c->rhs, nullptr, kOpaque);
    }
    ExprVisitor::VisitExpr_(op);
    this->AddNode(op);
  }
};

IndexedForwardGraph IndexedForwardGraph::Create(support::Arena* arena, const Expr& body) {
  return Creator(arena).Prepare(body);
}

/*!
 * \brief Dominator tree that represent domination or
 *  post domination relation of the node.
 */
class DominatorTree {
 public:
  /*!
   * \brief A node in the dominator tree.
   */
  struct Node {
    /*! \brief The node in the tree */
    IndexedForwardGraph::Node* gnode{nullptr};
    /*! \brief parent of the tree */
    Node* parent{nullptr};
    /*! \brief current depth*/
    int depth{0};
    /*! \brief aggregated pattern to parent */
    OpPatternKind pattern{kOpaque};
  };
  // index -> node.
  std::vector<Node*> nodes;
  /*!
   * \brief compute a post dominator relation for a given dataflow graph.
   * \param arena The arena used for node allocation.
   * \param graph The graph to be analyzed.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA (Least Common Ancestor)
   */
  static DominatorTree PostDom(support::Arena* arena, const IndexedForwardGraph& graph);

 private:
  // Combine pattern together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common ancestor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of the two.
   */
  static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern) {
    while (lhs != rhs) {
      if (lhs == nullptr) return nullptr;
      if (rhs == nullptr) return nullptr;
      if (lhs->depth < rhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        rhs = rhs->parent;
      } else if (rhs->depth < lhs->depth) {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        lhs = lhs->parent;
      } else {
        edge_pattern[0] = CombinePattern(edge_pattern[0], lhs->pattern);
        edge_pattern[0] = CombinePattern(edge_pattern[0], rhs->pattern);
        lhs = lhs->parent;
        rhs = rhs->parent;
      }
    }
    return lhs;
  }
  /*!
   * \brief Find the least common ancestor of a list of nodes.
   * \param nodes the nodes.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of all nodes.
   */
  Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes,
                            OpPatternKind* edge_pattern) {
    auto link = input_nodes.head;
    if (link == nullptr) {
      return nullptr;
    }
    auto get_node = [&](const IndexedForwardGraph::Edge& edge) {
      size_t oindex = edge.node->index;
      ICHECK_LT(oindex, nodes.size());
      Node* onode = nodes[oindex];
      ICHECK(onode != nullptr);
      return onode;
    };
    Node* parent = get_node(link->value);
    *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    link = link->next;
    for (; link != nullptr; link = link->next) {
      parent = LeastCommonAncestor(parent, get_node(link->value), edge_pattern);
      *edge_pattern = CombinePattern(*edge_pattern, link->value.pattern);
    }
    return parent;
  }
  /*!
   * \brief Convert the Node from an IndexedForwardGraph Node into DomaintorTree Node.
   * \param arena The Arena.
   * \param gnode An IndexedForwardGraph Node.
   * \return The DominatorTree Node.
   */
  Node* GetNode(support::Arena* arena, IndexedForwardGraph::Node* gnode) {
    Node* tnode = arena->make<Node>();
    tnode->gnode = gnode;
    if (gnode->extern_ref) {
      tnode->depth = 1;
      tnode->parent = nullptr;
      tnode->pattern = kOpaque;
    } else {
      // find the LCAs of all outputs.
      OpPatternKind pattern = kElemWise;
      Node* parent = LeastCommonAncestor(gnode->outputs, &pattern);
      tnode->depth = parent ? parent->depth + 1 : 1;
      tnode->parent = parent;
      tnode->pattern = pattern;
    }
    return tnode;
  }
};

DominatorTree DominatorTree::PostDom(support::Arena* arena, const IndexedForwardGraph& graph) {
  DominatorTree tree;
  tree.nodes.resize(graph.post_dfs_order.size(), nullptr);
  // reverse topo order
  for (size_t i = graph.post_dfs_order.size(); i != 0; --i) {
    size_t index = i - 1;
    tree.nodes[index] = tree.GetNode(arena, graph.post_dfs_order[index]);
  }
  return tree;
}

/*!
 * \brief A partition of the graph marked by union find data structure.
 */
class GraphPartitioner {
 public:
  explicit GraphPartitioner(support::Arena* arena, int opt_level, size_t max_fuse_depth,
                            bool is_custom_pass)
      : arena_(arena),
        opt_level_(opt_level),
        max_fuse_depth_(max_fuse_depth),
        is_custom_fusion_pass_(is_custom_pass) {}
  /*!
   * \brief Group as a union find data structure.
   */
  struct Group {
    /*! \brief The corresponding backend operator name. */
    std::string backend_op_name = kInvalidBackendOp;
    /*! \brief The parent in the union find data structure. */
    Group* parent{nullptr};
    /*! \brief The pattern of the group */
    OpPatternKind pattern;
    /*! \brief reference to the root node. */
    const tvm::Object* root_ref{nullptr};
    /*!
     * \brief Reference to the anchor node,
     * this field is not nullptr only if pattern is kOutEWiseFusable.
     */
    const tvm::Object* anchor_ref{nullptr};
    /*!
     * \brief Find the group root, perform path compression
     * \return The root type node.
     */
    Group* FindRoot() {
      // fast path
      if (this->parent == nullptr) return this;
      // slow path with path compression.
      Group* root = this;
      while (root->parent != nullptr) {
        root = root->parent;
      }
      for (Group* p = this; p != root;) {
        Group* parent = p->parent;
        p->parent = root;
        p = parent;
      }
      return root;
    }

    /*!
     * \brief The number of nodes belonging to this group
     */
    uint32_t num_nodes{1};
  };
  /*!
   * \brief Partition a graph.
   * \return group assignments of each node.
   */
  std::vector<Group*> Partition(const IndexedForwardGraph& graph);

 private:
  /*! \brief The map between op_name and . */
  std::unordered_map<int, IndexedForwardGraph::Node*> b_op_to_last_node_;
  /*! \brief Whether current op is from tensorrt or not to allow wider variety of fusion */
  bool is_tensorrt_op_ = false;

  /*! \brief The internal arena for temporary space. */
  support::Arena* arena_;
  /*! \brief optimization level for fuse operation. */
  int opt_level_;
  /*! \brief The maximum number of operations in one fused function */
  size_t max_fuse_depth_;
  /*! \brief Whether it is our custom fusion pass or not */
  bool is_custom_fusion_pass_ = false;
  /*! \brief The internal groups. */
  std::vector<Group*> groups_;
  /*! \brief internal field used for deduplication */
  std::unordered_set<IndexedForwardGraph::Node*> visited_;
  // Internal implelementation of CheckPath
  template <typename F>
  bool CheckPath_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    if (visited_.count(src)) return true;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    gnode = gnode->FindRoot();
    if (!fcond(gnode->pattern, src == sink)) return false;
    if (src == sink) return true;
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  /*!
   * \brief Check all the node and edge pattern
   *  between src and sink satisfies fcond.
   *
   * src is not checked.
   *
   * \param src The source node.
   * \param sink The termination node.
   * \param fcond The condition to be checked.
   * \tparam F the condition function, with signature
   * \note sink must be a post-dominator of src.
   */
  template <typename F>
  bool CheckPath(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, F fcond) {
    ICHECK(!src->extern_ref);
    visited_.clear();
    ICHECK(src != sink);
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      if (!CheckPath_(link->value.node, sink, fcond)) return false;
    }
    return true;
  }
  // Combine two patterns together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs,
                                      bool is_tensorrt = false) {
    // For custom fusion pass, if it is TensorRT op, we should allow fusion
    if (!is_tensorrt) {
      if (lhs > kBroadcast && rhs > kBroadcast) {
        LOG(FATAL) << "Cannot merge two complex group together";
      }
    }
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Merge the child group to the parent.
   * \param child The child group.
   * \param parent The parent group.
   */
  void MergeFromTo(Group* child, Group* parent) {
    child = child->FindRoot();
    parent = parent->FindRoot();
    if (child == parent) return;
    // update the number of nodes of the parent group
    parent->num_nodes += child->num_nodes;
    child->parent = parent;
    // update anchor ref and pattern
    if (child->anchor_ref != nullptr) {
      // For custom fusion pass, if it is TensorRT op, we should allow fusion
      if (!is_tensorrt_op_) ICHECK(parent->anchor_ref == nullptr);
      ICHECK(parent->anchor_ref == nullptr);
      parent->anchor_ref = child->anchor_ref;
      parent->pattern = CombinePattern(child->pattern, parent->pattern, is_tensorrt_op_);
    }
  }
  // Internal implelementation of CommitFuse
  void CommitFuse_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink, Group* target) {
    if (src == sink) return;
    if (visited_.count(src)) return;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    // merge the current group to the parent if possible.
    MergeFromTo(gnode, target);
    if (!is_custom_fusion_pass_) {
      for (auto link = src->outputs.head; link != nullptr; link = link->next) {
        CommitFuse_(link->value.node, sink, target);
      }
    }
  }
  /*!
   * \brief Commit fusion operation.
   * \param src The source node.
   * \param sink The termination node.
   * \note sink must be a post-dominator of src.
   */
  void CommitFuse(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
    Group* target = groups_[sink->index];
    visited_.clear();
    ICHECK(src != sink);
    CommitFuse_(src, sink, target);
  }

  size_t CountNodesUptoSink_(IndexedForwardGraph::Node* src, IndexedForwardGraph::Node* sink) {
    if (src == sink || visited_.count(src)) return 0;
    visited_.insert(src);
    Group* gnode = groups_[src->index];
    ICHECK(gnode != nullptr);
    auto sum = gnode->num_nodes;
    for (auto link = src->outputs.head; link != nullptr; link = link->next) {
      sum += CountNodesUptoSink_(link->value.node, sink);
    }
    return sum;
  }

  // Count the number of nodes in a fused subgraph if child is additionaly fused.
  // dom_parent is already known to be a part of the subgraph.
  // For a diamond structure, there can be multiple paths connecting child and dom_parent.
  // All intermediate nodes between child and dom_parent are taken into account.
  // Since dom_parent can itself be an intermediate node in the subgraph, calling FindRoot()
  // is important for correct calculation.
  size_t CountFusedNodesWithNewChild(IndexedForwardGraph::Node* child,
                                     IndexedForwardGraph::Node* dom_parent) {
    Group* target = groups_[dom_parent->index];
    visited_.clear();
    ICHECK(child != dom_parent);
    return target->FindRoot()->num_nodes + CountNodesUptoSink_(child, dom_parent);
  }

  // Initialize the groups.
  void InitGroups(const IndexedForwardGraph& graph) {
    groups_.resize(graph.post_dfs_order.size());
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      const auto* graph_node = graph.post_dfs_order[nid];
      auto* group_node = arena_->make<Group>();
      group_node->pattern = graph_node->pattern;
      group_node->root_ref = graph_node->ref;
      // set anchor ref if necessary.
      if (group_node->pattern == kOutEWiseFusable) {
        group_node->anchor_ref = graph_node->ref;
      }
      groups_[nid] = group_node;
    }
  }

  bool IsTensorRTOp(std::string backend_op_name) {
    int delim_pos = backend_op_name.find("_");
    std::string backend_name = backend_op_name.substr(0, delim_pos);
    bool is_tensorrt = false;
    if (backend_name == "tensorrt") is_tensorrt = true;

    return is_tensorrt;
  }

  // Fused based on backend operator matches from DP
  void RunFuseWithMap(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree) {
    VLOG_CONTEXT <<  "RunFuseWithMap";
    LOG(INFO) << "Running fusion with map";
    // WARNING(@Soo): We assume that fused ops are always continuous in the post dfs order.
    // THIS IS NOT TRUE, e.g., conv+add+relu for ResNet-50.
    //        std::cerr << "# of groups : " << groups_.size() << std::endl;

    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph.post_dfs_order[nid];
      Group* group_node = groups_[nid];
      ICHECK(group_node != nullptr);

      //          std::cerr << "Group node (" << nid << ") pattern: " << group_node->pattern <<
      //          std::endl;

      // Assign backend op name
      // WARNING(@Soo): We should assume that fused ops are not always opaque.
      //          const tvm::Object* cur_key = graph_node->ref;
      //          assert (graph.exprnode_to_backend_op.find(cur_key) !=
      //          graph.exprnode_to_backend_op.end()); std::cerr << "Expr: " <<
      //          GetRef<ObjectRef>(graph_node->ref) << std::endl;
      //          PrintOpType(GetRef<ObjectRef>(graph_node->ref));
      //          std::cerr << "backend: " << graph_node->backend << std::endl;
      ICHECK(graph_node->backend.compare("default") != 0)
          << "No backend was assigned for node " << nid << ": "
          << GetRef<ObjectRef>(graph_node->ref);
      GroupIdOpNamePair pair_info = GroupIdOpNamePair(graph_node->backend);
      group_node->backend_op_name = pair_info.backend_op_name;

      // Note that Var or Constant will be filtered out by this.
      // Softmax is also kOpaque
      if (group_node->pattern == kOpaque) continue;

      // Commit fuse if there was previous node
      // for correspding bakcend_op_id (= group_id)
      int cur_group_id = pair_info.group_id;
      if (b_op_to_last_node_.find(cur_group_id) != b_op_to_last_node_.end()) {
        auto* prev_graph_node = b_op_to_last_node_[cur_group_id];
        //            std::cerr << "-------------------------------------------------" << std::endl;
        //            std::cerr << "cur_group id: " << cur_group_id << ", " <<
        //                "Merge from "  << prev_graph_node->index << " to " << nid << std::endl;
        is_tensorrt_op_ = IsTensorRTOp(pair_info.backend_op_name);
        CommitFuse(prev_graph_node, graph_node);
      }
      b_op_to_last_node_[cur_group_id] = graph_node;
    }

    //        std::cerr << "------------------------------------" << std::endl;
    //        for (size_t nid = 0; nid < groups_.size(); ++nid) {
    //          Group* group_node = groups_[nid];
    //          std::cerr << "\tGroup " << nid << " (root_group): " <<
    //          GetRef<ObjectRef>(group_node->FindRoot()->root_ref) << std::endl;
    //        }
  }

  // execute the fusion algorithm.
  void RunFuse(const IndexedForwardGraph& graph, const DominatorTree& post_dom_tree, int phase) {
    for (size_t nid = 0; nid < groups_.size(); ++nid) {
      // the group of current node has been specified already.
      auto* graph_node = graph.post_dfs_order[nid];
      auto* dom_node = post_dom_tree.nodes[nid];

      CHECK_EQ(graph_node->ref, dom_node->gnode->ref);
      CHECK_EQ(graph_node, dom_node->gnode);

      Group* group_node = groups_[nid];
      ICHECK(group_node != nullptr);
      // no actions for opaque nodes
      if (group_node->pattern == kOpaque) continue;
      // no actions needed if the current node have no dominator
      if (dom_node->parent == nullptr) continue;
      ICHECK(!graph_node->extern_ref);
      size_t dom_parent_gindex = dom_node->parent->gnode->index;

      // refuse the fusion if too many ops are going to be fused together
      if (CountFusedNodesWithNewChild(graph_node, dom_node->parent->gnode) > max_fuse_depth_)
        continue;

      if (phase == 2) {
        // Fuse injective ops into intermediate tuples, if any
        if (group_node->pattern > kInjective) continue;
        Group* dom_parent_group = groups_[dom_parent_gindex];
        Group* dom_root_group = dom_parent_group->FindRoot();
        // If dom node group has a tuple as its root, we do not fuse tuple fields into it
        if (dom_root_group->pattern == kTuple) continue;
        if (dom_parent_group->pattern == kTuple && dom_root_group->pattern <= kInjective) {
          // Now we know the tuple has been fused into subsequent injective ops
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
          // dom_root_group can also be tuple, as in inception layers
          // CheckPath is needed to avoid fusing two intermediate tuples
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
        continue;
      }

      // Skip if current node is already fused to the parent.
      if (groups_[dom_parent_gindex] != nullptr &&
          group_node->FindRoot() == groups_[dom_parent_gindex]->FindRoot()) {
        continue;
      }
      // Do not fuse into tuple for now
      if (groups_[dom_parent_gindex]->pattern == kTuple) continue;
      // Try to fuse current node to its post-dominator.
      if (group_node->pattern == kOutEWiseFusable) {
        if (phase != 0) continue;
        // Path for OutEWiseFusable: conv2d
        // Check if the dominator relation is elemwise.
        if (dom_node->parent != nullptr && dom_node->pattern == kElemWise) {
          ICHECK(dom_node->parent->gnode != nullptr);
          // The fuse can be executed if all the intermediate ops are still broadcast.
          auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kBroadcast; };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern <= kBroadcast) {
        // Pre-condition: can only be fused to parent which is injective or reduction.
        if (dom_node->parent != nullptr &&
            (dom_node->pattern <= kInjective || dom_node->pattern == kCommReduce)) {
          // Check if all the intermediate ops are still broadcast.
          // The final terminal node can already be fused to a OutEWiseFusable group.
          auto fcond = [](OpPatternKind kind, bool is_sink) {
            if (!is_sink) {
              // Elemwise, broadcast, and injective ops on the parallel branches
              // are allowed be fused to the elemwise/broadcast anchor.
              return kind <= kInjective;
            } else {
              return (kind <= kBroadcast || kind == kCommReduce || kind == kInjective ||
                      kind == kOutEWiseFusable);
            }
          };
          if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
            CommitFuse(graph_node, dom_node->parent->gnode);
          }
        }
      } else if (group_node->pattern == kInjective || group_node->pattern == kTuple) {
        // defer injective fusion to second phase.
        // so conv2d always finishes fusing.
        if (phase != 1) continue;
        // Check if all path are injective.
        auto fcond = [](OpPatternKind kind, bool is_sink) { return kind <= kInjective; };
        if (CheckPath(graph_node, dom_node->parent->gnode, fcond)) {
          CommitFuse(graph_node, dom_node->parent->gnode);
        }
      } else {
        // do nothing.
        ICHECK(group_node->pattern == kCommReduce);
      }
    }
  }
};

std::vector<GraphPartitioner::Group*> GraphPartitioner::Partition(
    const IndexedForwardGraph& graph) {
  this->InitGroups(graph);
  if (opt_level_ == 0) return std::move(groups_);
  // get post dominator tree
  auto post_dom_tree = DominatorTree::PostDom(arena_, graph);

  // If we don't use DP, execute original TVM fusion
  if (!is_custom_fusion_pass_) {
    // std::cerr << "original fusion pass" << std::endl;
    //  run fusion algorithm.
    for (int phase = 0; phase < 3; ++phase) {
      this->RunFuse(graph, post_dom_tree, phase);
    }
  } else {
    //        std::cerr << "Custom fusion pass" << std::endl;
    this->RunFuseWithMap(graph, post_dom_tree);
  }

  return std::move(groups_);
}

class FuseMutator : private MixedModeMutator {
 public:
  // Run the transform
  Expr Transform(const Expr& body, int fuse_opt_level, size_t max_fuse_depth,
                 bool is_custom_pass = false) {
    // setup the group map.
    auto graph = IndexedForwardGraph::Create(&arena_, body);
    auto groups =
        GraphPartitioner(&arena_, fuse_opt_level, max_fuse_depth, is_custom_pass).Partition(graph);
    for (size_t nid = 0; nid < graph.post_dfs_order.size(); ++nid) {
      ICHECK(graph.post_dfs_order[nid]->ref != nullptr);
      gmap_[graph.post_dfs_order[nid]->ref] = groups[nid];
    }
    // The following line can be used for debug.
    // this->DebugDumpGroup(body);
    return this->Mutate(body);
  }

 private:
  using MixedModeMutator::VisitExpr_;

  /*! \brief Temporary information from each group. */
  struct GroupInfo {
   public:
    // The parameters of the function.
    Array<Var> params;
    // The arguments to call the functions.
    Array<Expr> arguments;
    // Get a new parameter or allocate an old one
    Var GetOrAllocParam(const Expr& expr, const Type& type) {
      // run linear scan as most fused groups contain only a few inputs.
      for (size_t i = 0; i < arguments.size(); ++i) {
        if (expr.same_as(arguments[i])) return params[i];
      }
      // create a new parameter.
      std::ostringstream os;
      os << "p" << params.size();
      auto var = Var(os.str(), type);
      params.push_back(var);
      arguments.push_back(expr);
      return var;
    }
  };
  /*! \brief Internal arena. */
  support::Arena arena_;
  /*! \brief The group assignment map. */
  std::unordered_map<const Object*, GraphPartitioner::Group*> gmap_;
  /* \brief Internal group information map. */
  std::unordered_map<GraphPartitioner::Group*, GroupInfo> ginfo_;

  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    if (fn_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(fn_node);
    } else {
      return ExprMutator::VisitExpr_(fn_node);
    }
  }

  // Transform calls.
  Expr Rewrite_(const CallNode* call, const Expr& post) {
    if (call->op.as<OpNode>()) {
      static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");

      if (fnoncomputational.get(Downcast<Op>(call->op), false)) {
        return ExprMutator::VisitExpr_(call);
      }

      // If it is a primitive op call
      // then we must have a group assignment for it already.
      ICHECK(gmap_.count(call));
      if (call->op == stop_fusion_op) {
        return ExprMutator::VisitExpr(call->args[0]);
      }
      auto* ret_group = gmap_.at(call)->FindRoot();
      Array<Expr> new_args = GetNewArguments(call->args, ret_group);

      auto new_call = Call(call->op, new_args, call->attrs, call->type_args, call->span);

      if (ret_group->root_ref == call) {
        // This is the root of the group
        // create the new call node.
        return MakeNewFunction(ret_group, call->checked_type(), new_call);
      } else {
        // This is an intermediate node of a fused function
        // simply return the new call.
        return std::move(new_call);
      }
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) {
    auto* ret_group = gmap_.at(tuple_node)->FindRoot();
    if (ret_group->root_ref == tuple_node) {
      return ExprMutator::VisitExpr_(tuple_node);
    }
    // This tuple is an intermediate node in the group
    Array<Expr> new_fields = GetNewArguments(tuple_node->fields, ret_group);
    return Tuple(new_fields);  // TODO(mbs): No backend copy
  }

  Expr Rewrite_(const TupleGetItemNode* tuple_get, const Expr& post) {
    auto* ret_group = gmap_.at(tuple_get)->FindRoot();
    auto new_tuple = GetNewArguments({tuple_get->tuple}, ret_group)[0];
    auto new_node = TupleGetItem(new_tuple, tuple_get->index);
    if (ret_group->root_ref == tuple_get) {
      if (gmap_.at(tuple_get->tuple.get())->FindRoot() != ret_group) {
        // Isolated. This case occurs when tuple is created by an Opaque op
        // e.g. multibox_transform_loc
        return ExprMutator::VisitExpr_(tuple_get);
      }
      // A new function whose output is a tuple field access
      return MakeNewFunction(ret_group, tuple_get->checked_type(), new_node);
    }
    // This is an intermediate node in the group
    return std::move(new_node);
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      // Visit body and cache the op
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        this->memo_[expr] = expr;
      } else {
        this->memo_[expr] = Let(var, value, body);
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr MakeNewFunction(GraphPartitioner::Group* group, Type ret_type, Expr body) {
    // Quickly check special properties of the fused function.
    // A pass to check if the fused op contains only reshape ops.
    class CheckReshapeOnly : public ExprVisitor {
     public:
      void VisitExpr_(const CallNode* cn) final {
        this->has_call = true;
        static auto freshape_op = Op::GetAttrMap<TReshapeOp>("TReshapeOp");

        if (!freshape_op.get(cn->op, false)) {
          this->reshape_only = false;
        }

        if (!this->reshape_only) return;
        ExprVisitor::VisitExpr_(cn);
      }

      void VisitExpr_(const VarNode* vn) final {
        if (!vn->type_annotation.defined() || !vn->type_annotation->IsInstance<TensorTypeNode>()) {
          this->reshape_only = false;
        }
      }

      bool reshape_only = true;
      bool has_call = false;
    } visitor;

    visitor(body);
    const GroupInfo& ginfo = ginfo_[group];
    auto func = Function(ginfo.params, body, ret_type, {});
    func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(visitor.has_call));
    // TODO(mbs): "reshape" cleanup.
    if (visitor.has_call && visitor.reshape_only) {
      func = WithAttr(std::move(func), attr::kReshapeOnly, tvm::Integer(visitor.reshape_only));
    }
    // PATCH(@Soo): Add backend op attribute.
    func = WithAttr(std::move(func), attr::kBackendOp, String(group->backend_op_name));
    return Call(func, ginfo.arguments, Attrs());
  }

  Array<Expr> GetNewArguments(const tvm::Array<Expr>& args,
                              GraphPartitioner::Group* current_group) {
    Array<Expr> new_args;
    for (auto arg : args) {
      auto* arg_group = gmap_.at(arg.get())->FindRoot();
      auto type = arg->checked_type();
      Expr new_arg = this->Mutate(arg);
      if (current_group != arg_group) {
        Var param = ginfo_[current_group].GetOrAllocParam(new_arg, type);
        new_args.push_back(param);
      } else {
        new_args.push_back(new_arg);
      }
    }
    return new_args;
  }

  // Debug function, dump the group assignment in text.
  void DebugDumpGroup(const Expr& body) {
    std::string text = AsText(body, false, [this](const ObjectRef& expr) -> std::string {
      auto it = gmap_.find(expr.get());
      if (it == gmap_.end()) return "";
      std::ostringstream os;
      auto* group = it->second->FindRoot();
      os << " /* group=" << group << " */";
      return os.str();
    });
    LOG(INFO) << "Dump of group info:\n" << text;
  }
};

// For op measurements, execute original fusion pass
// For end-to-end measure, execute user-defined pass by using best match
// from PlanFusionWithExtCompiler
Expr FuseOps(const Expr& expr, int fuse_opt_level, size_t max_fuse_depth, const IRModule& module) {
  // Debug: visualization
  // auto vis_call = tvm::runtime::Registry::Get("collage.optimizer.visualize_expr");
  // (*vis_call)(expr, "before_fusion");

  // WARNING(@Soo): Assume that all exprs are function!
  const auto* fn_node = expr.as<FunctionNode>();
  ICHECK(fn_node);
  bool is_custom_pass = false;
  // Warning(@Soo) - every fusion should be done by saved match log regardless of algorithms!
  // If you comment this line, we can try original fusion pass.
  if (fn_node->GetAttr<IntImm>(attr::kCustomFusionPass).defined()) is_custom_pass = true;
  auto fused_expr = FuseMutator().Transform(expr, fuse_opt_level, max_fuse_depth, is_custom_pass);
  //      std::cerr << "[Done] FuseOps (is_custom_pass: " << is_custom_pass << ")" << std::endl;

  // Debug: visualization
  // std::string network_name = std::string(fn_node->GetAttr<String>("Network").value());
  // auto vis_call2 = tvm::runtime::Registry::Get("collage.optimizer.visualize_expr");
  // (*vis_call2)(fused_expr, network_name+"_after_fusion");
  // (*vis_call2)(fused_expr, "after_fusion");

  return fused_expr;
}

Expr InferBackendForConstantFunc(const Expr& expr) {
  // WARNING(@Soo): Assume that all exprs are function!
  const auto* fn_node = expr.as<FunctionNode>();
  ICHECK(fn_node);
  bool is_custom_pass = false;

  // Warning(@Soo) - every fusion should be done by saved match log regardless of algorithms!
  // If you comment this line, we can try original fusion pass.
  if (fn_node->GetAttr<IntImm>(attr::kCustomFusionPass).defined()) is_custom_pass = true;

  // Warning(@Soo): It only works for CHAIN FUSION PATTERN now
  struct AssignNewOpGroupToBrokenChildren : ExprVisitor {
    std::string parent_of_new_node_backend;
    bool is_any_broken_children = false;

    void Assign(std::string backend_name, const Expr& expr) {
      is_any_broken_children = false;
      parent_of_new_node_backend = backend_name;

      this->VisitExpr(expr);
      if (is_any_broken_children) ++NEW_BACKEND_GROUP_ID_FUSE_PASS;
    }

    void VisitExpr_(const CallNode* op) final {
      if (op->backend().operator std::string().compare(parent_of_new_node_backend) == 0) {
        is_any_broken_children = true;
        // This needs to be updated to afford all other backends
        // Not only the autotvm
        std::string new_backend = std::to_string(NEW_BACKEND_GROUP_ID_FUSE_PASS) + "-autotvm";
        GetRef<Expr>(op).set_backend(new_backend);
        for (auto arg : op->args) {
          this->VisitExpr(arg);
        }
      }
    }
  };

  struct InferBackendForConstantVisitor : ExprVisitor {
    String parent_backend_ = "default";

    void VisitExpr_(const CallNode* op) final {
      this->VisitSpan(op->span);
      this->VisitExpr(op->op);

      for (auto ty_arg : op->type_args) {
        this->VisitType(ty_arg);
      }

      // Change backend if it is default and the op is "expand_dims"
      // Let's block other ops than expand_dims to prevent side effects
      if (op->backend().operator std::string().compare("default") == 0) {
        if (const OpNode* op_node = op->op.as<OpNode>()) {
          if (op_node->name == "expand_dims") {
            GetRef<Expr>(op).set_backend(parent_backend_);
          } else if (op_node->name == "layout_transform") {
            // If layout_transform is a parent of Call, then we shouldn't fuse it with
            // the following Call function; that's what TVM does.
            // It is also faster without fusion.
            // e.g., in one convolution test, it seems that fusion is slower than non-fusion case.
            UpdateBackendWithNewGroup<CallNode>(GetRef<Expr>(op));

            // If you want fusion with child or parent, use the code below
            //                const CallNode* child_call = op->args[0].as<CallNode>();
            //                auto child_backend = child_call->backend;
            //                MutateBackendCopy(GetRef<Expr>(op), child_backend);
            //                MutateBackendCopy(GetRef<Expr>(op), parent_backend_);

            // Create new group id to children fused ops that are broken bu layout_transform
            // Warning(@Soo): It only works for CHAIN FUSION PATTERN now
            // We can safely assume that the input of layout transformation is always call node.
            AssignNewOpGroupToBrokenChildren().Assign(parent_backend_.operator std::string(),
                                                      op->args[0]);
          } else {
            ICHECK(0) << "Unexpected operator type (" << op_node->name << ") "
                      << "with the backend of default"
                      << "It is likely that this op was changed in AlterOpLayout";
          }
        }
      }
      //          std::cerr << "Parent backend : " << op->backend.operator std::string() <<
      //          std::endl;
      for (auto arg : op->args) {
        parent_backend_ = op->backend();
        this->VisitExpr(arg);
      }
    }

    void VisitExpr_(const ConstantNode* op) final {
      //          std::cerr << "Mutate backend : " << parent_backend_.operator std::string() <<
      //          std::endl;
      GetRef<Expr>(op).set_backend(parent_backend_);
      ExprVisitor::VisitExpr_(op);
    }
  } visitor;

  if (is_custom_pass) visitor(expr);
  //      std::cerr << "[InferBackendForConstant] " << expr << std::endl;
  //      std::cerr << "[Done] InferBackendForConstant" << std::endl;
  return expr;
}

IRModule VisualizeIRFunc(IRModule module, String filename = "default") {
  Expr expr = module->Lookup("main");
  const auto* vis_call = tvm::runtime::Registry::Get("collage.optimizer.visualize_network_debug");
  ICHECK(vis_call);
  (*vis_call)(expr, filename);
  return module;
}

/*
 * Goal
 * - Calculate best opeartor match depending on which algorithm we pick.
 * - Dump best match into the log and let FuseOps pass read and apply it on IR.
 */
IRModule AssignBackendFunc(IRModule module) {
  //      std::cerr << "Plan Fusion With ExtCompiler" << std::endl;
  Expr expr = module->Lookup("main");
  const FunctionNode* fn_node = expr.as<FunctionNode>();
  ICHECK(fn_node);

  // PATCH(@Soo): New custom fusion pass type
  constexpr int kUserDefinedFusion = 0;
  constexpr int kDP = 1;
  constexpr int kExhaustiveSearch = 2;
  constexpr int kTwoLevelOpt = 3;
  constexpr int kOpMeasurement = 4;
  constexpr int kSingleBackendBaseline = 5;
  constexpr int kVisualizeBackendPlacement = 6;

  // Do nothing when it's not custom fusion pass
  if (fn_node->GetAttr<IntImm>(attr::kCustomFusionPass).defined()) {
    int64_t custom_fusion_pass_type =
        fn_node->GetAttr<IntImm>(attr::kCustomFusionPass).as<IntImmNode>()->value;
    std::string custom_fusion_pass_str;
    // PATCH(@Soo): Custom (DP) fusion pass for user defined fusion
    if (custom_fusion_pass_type == kUserDefinedFusion) {
      custom_fusion_pass_str = "collage.optimizer.get_user_fusion";
      // PATCH(@Soo): Custom (DP) fusion pass for subprocess call during the end-to-end measurements
      // Note that if fuse_opt_level == 0, no fusion applied no matter whether it's original or DP.
    } else if (custom_fusion_pass_type == kDP) {
      custom_fusion_pass_str = "collage.optimizer.run_dp";
    } else if (custom_fusion_pass_type == kTwoLevelOpt) {
      custom_fusion_pass_str = "collage.optimizer.run_two_level_opt";
    } else if (custom_fusion_pass_type == kExhaustiveSearch) {
      custom_fusion_pass_str = "collage.optimizer.run_exhaustive_search";
    } else if (custom_fusion_pass_type == kOpMeasurement) {
      custom_fusion_pass_str = "collage.optimizer.assign_backend_for_op_measurement";
    } else if (custom_fusion_pass_type == kSingleBackendBaseline) {
      custom_fusion_pass_str = "collage.optimizer.run_single_backend_baseline";
    } else if (custom_fusion_pass_type == kVisualizeBackendPlacement) {
      custom_fusion_pass_str = "collage.optimizer.visualize_backend_placement";
    } else {
      ICHECK(false) << "Fusion pass type " << fn_node->GetAttr<IntImm>(attr::kCustomFusionPass)
                    << "is not expected\n\n";
    }

    LOG(INFO) << "Invoking " << custom_fusion_pass_str;
    const auto* fdp_call = tvm::runtime::Registry::Get(custom_fusion_pass_str);
    ICHECK(fdp_call);
    // Note that we don't need this match
    // because we dump this into the file and will load it for backend operator decision
    // This is to prevent segmentation fault; still, we don't know whether this helps
    (*fdp_call)(expr);
    LOG(INFO) << "Done with " << custom_fusion_pass_str;
    //        if (custom_fusion_pass_type != kUserDefinedFusion) {
    //          Map<Expr, String> backend_op_match = (*fdp_call)(expr);
    //        }
    //        std::cerr << "Before extenral pass: " << expr << std::endl;

    // Visualization of the network for sanity check
    //        auto vis_call =
    //        tvm::runtime::Registry::Get("collage.optimizer.visualize_network_debug"); Expr expr =
    //        module->Lookup("main");
    //        (*vis_call)(expr, "before_AssignTensorRT");
    //        std::cerr << "[Done] Debug visualization" << std::endl;

    // Apply external compiler ops first before we fuse operators
    // just like what original TensorRT pipeline does.
    LOG(INFO) << "Invoking " << "collage.optimizer.apply_external_compiler_op";
    const auto* ex_op_call = tvm::runtime::Registry::Get("collage.optimizer.apply_external_compiler_op");
    ICHECK(ex_op_call);
    // Warning(@Soo): Doublecheck if module is updated.
    module = (*ex_op_call)(module);
    LOG(INFO) << "Done with " << "collage.optimizer.apply_external_compiler_op";
    //        std::cerr << "[Done] PlanFusionWithExtCompiler" << std::endl;
  }

  //      std::cerr << "[Done] AlterOpLayout for TVM ops after PlanFusionWithExtCompiler" <<
  //      std::endl;
  return module;
}

namespace transform {
Pass InferBackendForConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> infer_backend_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(InferBackendForConstantFunc(f));
      };

  return CreateFunctionPass(infer_backend_func, 1, "InferBackendForConstant", {});
}

Pass VisualizeIR(String filename) {
  // Custom Module pass to deal with external compiler, e.g., tensorrt
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> visualize_ir_func =
      [=](IRModule m, PassContext pc) { return VisualizeIRFunc(m, filename); };
  return CreateModulePass(visualize_ir_func, 0, "VisualizeIR", {});
}

Pass AssignBackend() {
  // Custom Module pass to deal with external compiler, e.g., tensorrt
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> assign_backend_func =
      [=](IRModule m, PassContext pc) { return AssignBackendFunc(m); };
  return CreateModulePass(assign_backend_func, 0, "AssignBackend", {});
}

TVM_REGISTER_GLOBAL("relay._transform.AssignBackend").set_body_typed(AssignBackend);

Pass FuseOps(int fuse_opt_level) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig("relay.FuseOps.max_depth", Integer(kMaxFusedOps));
        return Downcast<Function>(FuseOps(f, opt_level, max_fuse_depth.value(), m));
      };
  return CreateFunctionPass(pass_func, 0, "FuseOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseOps").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
