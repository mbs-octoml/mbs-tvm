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
 * \file src/relay/collage/sub_graph.h
 * \brief Represents a sub-graph of an overall Relay expression.
 */

#ifndef SRC_RELAY_COLLAGE_SUB_GRAPH_H_
#define SRC_RELAY_COLLAGE_SUB_GRAPH_H_

#include <tvm/relay/op_attr_types.h>

#include <string>

#include "../ir/dataflow_matcher_impl.h"
#include "../ir/indexed_graph.h"
#include "index_set.h"

namespace tvm {
namespace relay {
namespace collage {

/*! \brief Returns operator pattern kind as single-letter string. */
std::string KindToString(OpPatternKind kind);

/*!
 * \brief Returns a kind and label for the single \p sub_expr, ignoring it's sub-sub expressions.
 */
std::pair<OpPatternKind, std::string> SubExprKindAndLabel(const Expr& sub_expr);

/*!
 * \brief Returns a kind and label for all the nodes in \p inside.
 */
std::pair<OpPatternKind, std::string> SubGraphKindAndLabel(const DataflowGraph& dataflow_graph,
                                                           const IndexSet& inside);

/*!
 * \brief Returns the index set representing all the sub-expression matched by \p matcher.
 */
IndexSet MatcherToIndexSet(const DFPatternMatcher& matcher);

/*!
 * \brief Configuration controlling which sub-graphs are considered valid.
 */
struct SubGraphConfig {
  /*! \brief Maximum number of outputs from the sub-graph, or zero if no limits. */
  size_t max_outputs = 0;
  /*!
   * \brief Whether a node inside the sub-graph may flow to nodes both inside and outside
   * the sub-graph. Note that it is still possible to have multiple outputs even with this
   * flag false.
   */
  bool allow_taps = false;
  /*!
   * \brief Maximum allowed maximum depth, or zero if no-limit.
   */
  size_t max_max_depth = 0;
};

using LabelledSubGraphMap = Map<String, ObjectRef /* actually SubGraph */>;

/*!
 * \brief A compact representation of a sub-graph within an (implied) overall Relay expression
 * (typically the "main" Relay global function). Sub-graphs can be used to represent composite and
 * fused functions without having to pay the cost of constructing either the function or the
 * rewritten overall expression calling that function. We also allow functions to be extracted
 * independently of rewriting the overall expression since measuring candidate kernel performance
 * does not need the latter.
 *
 * A sub-graph classifies every sub-expression node of the overall expression as either 'inside' or
 * 'outside' the sub-graph. Not all such divisions make sense: see the \p IsValid method for the
 * rules, and \p SubGraphConfig for how to customize those rules.
 *
 * We use post-dfs expression node indexes to uniquely refer to expression nodes.
 * We expect O(thousands) of sub-graphs to be in flight for a given model, and avoid creating
 * fused function and partitioned graphs unless absolutely necessary for downstream processing.
 * Most operations require a \p DataflowGraph and possibly the overall expression, and those
 * are not kept within the sub-graph to save space.
 *
 * As well as 'inside' and 'outside' we have four other flavors of sub-expression node:
 *  - 'entry' nodes are those inside with at least one dataflow input outside.
 *  - 'exit' nodes are  those inside with at least one dataflow output outside, or which
 *    are considered 'external' in the underlying dataflow graph (eg because they represent
 *    the result of the overall function).
 *  - 'input' nodes are those outside with at least one dataflow output inside.
 *  - 'output' nodes are those outside with at least one dataflow input inside.
 * It is valid to have multiple entry nodes (we'll bind a parameter for each). It may be valid to
 * have multiple exit nodes (we'll build a tuple of all such). It may be valid to have exit nodes
 * which also contribute to other inside nodes (ie represent a 'top' on an intermediate result).
 *
 * Sub-graphs are closed under:
 *  - Disjoint union.
 *  - Wrapping by a label, which indicates the wrapped sub-graph should be extracted as a
 *    sub-function with a "Composite" label.
 *  - Substitution, which allows a sub-graph w.r.t. one dataflow graph to be transformed to
 *    match some other (typcially smaller) dataflow graph.
 *
 * See the subclasses of \p FusionRule for how sub-graphs are built and combined.
 */
class SubGraphNode : public Object {
 public:
  /*!
   * \brief Which sub-expressions are inside the sub-graph (using their post-dfs indexes w.r.t.
   * the implied DataflowGraph).
   */
  IndexSet inside_;

  /*!
   * \brief Index of first inside node.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  PostDfsIndex first_inside_index_;

  /*!
   * \brief Which sub-expressions are entry/exit/input/output for this sub-graph.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  IndexSet entry_;
  IndexSet exit_;
  IndexSet input_;
  IndexSet output_;

  /*!
   * \brief Maximum depth of any dataflow path from an entry to an output sub-expression.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  size_t max_depth_ = 0;

  /*!
   * \brief The \p OpPatternKind summarizing the input/output behavior of the sub-graph.
   *
   * A sub-graph consisting of a single Relay expression node is given kind:
   *  - For Call to a Relay operator, the "TOpPattern" attribute of that operator (provided the
   *    call does not involve data-dependent dynamic shapes).
   *  - For Call to Relay Function, the "TOpPattern" attribute of the function (provided it has
   *    that attribute)
   *  - For Constants, \p kElemWise.
   *  - For Tuple and tuple projections, \p kInjective (provided all tuple fields are of tensor
   *    type)
   *  - All other nodes \p kOpaque.
   * Sub-graphs with more than one node have the maximum of the kind of each node.
   *
   * Cached for performance, uniquely determined by inside_.
   */
  OpPatternKind kind_ = kOpaque;

  /*!
   * \brief A label for the sub-graph. Not guaranteed to be unique, but is a human-readable summary
   * of the sub-graph which can help with debugging.
   */
  String label_;

  /*!
   * \brief Maps labels to non-overlapping subsets of this sub-graph. It is valid for nodes inside
   * this sub-graph to not appear in any labelled sub-sub-graph.
   *
   * When constructing the fused function representing this sub-graph, we:
   *  - construct the fused sub-functions for all sub-sub-graphs.
   *  - attach labels as "Composite" attributes to those fused sub-functions.
   *  - construct the overall fused function, inserting calls to the above sub-functions where
   *    required.
   * In this we we can represent partitions for the purposes of labelling 'composite operators'
   * (eg as used by BYOC pattern-based integrations) inside partitions for the purpose of
   * representing a kernel. In principle the nesting could go even deeper.
   *
   * Range in this map is an ObjectRef since we can't forward reference (sigh).
   */
  LabelledSubGraphMap sub_sub_graphs_;

  // TODO(mbs): 'Anchor nodes' and rules for unioning them.
  // In FuseOps it's just the unique kEWiseFusable node, if any.
  // I'd like to allow writing vertical fusion rules, eg if two candidates are directly
  // connected and have nn.conv2d anchors allow their join.
  // I'd also like to allow horizontal fusion rules, eg if two candidates are not directly
  // connected but could be joined without producing invalid (eg cyclic) and have nn.conv2d anchors
  // then do so. Come back to this.

  /*! \brief Number of nodes in overall dataflow graph. */
  size_t overall_size() const { return inside_.end_index(); }

  bool IsEmpty() const { return inside_.IsZero(); };

  /*! \brief Number of nodes in sub-graph. */
  size_t Size() const { return inside_.PopCount(); }

  /*!
   * \brief Returns true if this sub-graph is valid. Ie:
   *  - no output of the sub-graph can flow to any input of the sub-graph (otherwise we'd end up
   *    with a dataflow cycle when we partition).
   *  - all inputs and outputs of the sub-graph are in the same scope, ie not separated by
   *    control flow (otherwise there'd be no consistent program point at which to eval the
   *    partitioned function).
   *  - no more than config.max_outputs outputs are require.
   *  - if config.allow_taps is false, no inside node has outputs to nodes both inside and
   *    outside the sub-graph.
   */
  bool IsValid(const DataflowGraph& dataflow_graph, const SubGraphConfig& config) const;

  /*!
   * \brief Returns a \p Function representing this sub-graph within overall expression represented
   * by \p dataflow_graph. The function will be assigned the given \p attrs. Note that the overall
   * expression is not rewritten to call the resulting function.
   */
  Function ExtractFunction(const DataflowGraph& dataflow_graph) const;

  /*!
   * \brief Returns \p expr (with matching \p dataflow_graph) rewritten to partition on this
   * sub-graph. If \p function_override is defined, use that as the extracted function, otherwise
   * use the function which would be returned by \p ExtractFunction above. Obviously if
   * \p function_override is given it must match the \p ExtractFunction function exactly on
   * argumens, types etc.
   */
  Expr Partition(const DataflowGraph& dataflow_graph, const Expr& expr,
                 const Function& function_override) const;

  std::string ToString() const;

  bool operator==(const SubGraphNode& that) const;
  bool operator!=(const SubGraphNode& that) const { return !(*this == that); }
  size_t hash() const;

 private:
  /*! \brief Initialize the entry/exit/input/output sets given the inside and \p dataflow_graph. */
  void Init(const DataflowGraph& dataflow_graph);

  /*! \brief Calculates and returns the maximum path depth. */
  size_t MaxDepth(const DataflowGraph& dataflow_graph) const;

  /*! \brief Return's true if any (input/output) of node is (outside/inside) the sub-graph. */
  bool AnyInputOutside(const DataflowGraph::Node* node) const;
  bool AnyInputInside(const DataflowGraph::Node* node) const;
  bool AnyOutputOutside(const DataflowGraph::Node* node) const;
  bool AnyOutputInside(const DataflowGraph::Node* node) const;

 public:
  static constexpr const char* _type_key = "relay.collage.SubGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubGraphNode, Object);

  friend class SubGraph;
};

class SubGraph : public ObjectRef {
 public:
  /*! \brief Primitive constructor. The following constructors are generally more convenient. */
  SubGraph(const DataflowGraph& dataflow_graph, IndexSet inside, OpPatternKind kind = kOpaque,
           String label = {}, LabelledSubGraphMap sub_sub_graphs = {});

  /*! \brief Constructs the empty sub-graph for \p dataflow_graph. */
  explicit SubGraph(const DataflowGraph& dataflow_graph);

  /*!
   * Returns true if this and that are disjoint.
   */
  bool AreDisjoint(const SubGraph& that) const;

  /*!
   * \brief Returns disjoint union of this and \p that sub-graphs. The result may not be valid.
   */
  SubGraph DisjointUnion(const DataflowGraph& dataflow_graph, const SubGraph& that) const;

  /*!
   * \brief Returns copy of this sub-graph with all nodes labeled by \p label.
   */
  SubGraph WithLabel(const DataflowGraph& dataflow_graph, String label) const;

  /*!
   * \brief Returns copy of this sub-graph with all indexes substituted according to \p subst,
   * whose range is w.r.t. \p new_dataflow_graph.
   */
  SubGraph Subst(const DataflowGraph& new_dataflow_graph,
                 const std::unordered_map<PostDfsIndex, PostDfsIndex>& subst) const;

  TVM_DEFINE_OBJECT_REF_METHODS(SubGraph, ObjectRef, SubGraphNode);
};

struct SubGraphEqual {
  bool operator()(const SubGraph& left, const SubGraph& right) const {
    return *left.get() == *right.get();
  }
};

struct SubGraphHash {
  size_t operator()(const SubGraph& sub_graph) const { return sub_graph->hash(); }
};

/*!
 * \brief Pass to partition every global function according to the post-dfs indexes
 * given in an array. Visible for testing from Python only, would never make sense to use
 * as a generic pass!
 */
transform::Pass PartitionOnIndexesForTesting(Array<Integer> indexes);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_SUB_GRAPH_H_
