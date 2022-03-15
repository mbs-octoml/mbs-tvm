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
 * \file src/relay/collage/sub_graph.cc
 * \brief Represents a sub-graph of an overall Relay expression.
 */

#include "./sub_graph.h"

#include <tvm/relay/transform.h>

#include "../transforms/pass_utils.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {

class Extractor;

/*!
 * \brief Helper class for rewriting expression to call (and possibly project) the extracted
 * function.
 */
class Rewriter : public ExprMutator {
 public:
  Rewriter(const Extractor* extractor) : extractor_(extractor) {}

  Expr VisitExpr(const Expr& expr) override;

  /*!
   * \brief Returns a substitution from indexes of the original expression to indexes of the
   * rewritten expression, given the \p new_dataflow_graph for the new expression.
   */
  IndexSubst MakeIndexSubst(const DataflowGraph& new_dataflow_graph) const;

 private:
  /*! \brief Already prepared extractor from which to take the call and output indexes. */
  const Extractor* extractor_;
};

/*! \brief Helper class for extracting matched sub-graphs from the overall expression. */
class Extractor : public ExprMutator {
 public:
  Extractor(const DataflowGraph* dataflow_graph, const SubGraphNode* sub_graph, String label,
            Function function_override)
      : dataflow_graph_(dataflow_graph),
        sub_graph_(sub_graph),
        label_(std::move(label)),
        function_override_(std::move(function_override)) {
    ICHECK_EQ(dataflow_graph_->size(), sub_graph_->overall_size());
  }

  const DataflowGraph& dataflow_graph() const { return *dataflow_graph_; }

  /*!
   * \brief Collect the parameters and output expressions for the function representing
   * the sub-graph. The result is available in:
   *  - num_outputs()
   *  - function()
   *  - call()
   *  - output_substitution()
   */
  void Extract() {
    ICHECK(!sub_graph_->IsEmpty());
    VLOG(2) << "Extracting " << sub_graph_->ToString();

    //  In reverse dataflow order...
    for (PostDfsIndex i = dataflow_graph_->size(); i > 0; --i) {
      PostDfsIndex index = i - 1;
      if (!sub_graph_->inside_[index]) {
        // Node is outside sub-graph.
        continue;
      }
      VLOG(2) << "index " << index;
      auto node = dataflow_graph_->index_to_node(index);
      if (sub_graph_->exit_[node->index_] || node->is_external_ || memo_.count(node->ref()) == 0) {
        // This sub-expression is:
        //  - inside the sub-graph and needed outside the sub-graph. So it must contribute to the
        //    output of the function (even if we've already visited it while constructing an
        //    output from a downstream sub-expression).
        //  - not yet visited, in which case we must evaluate it for its side effects.
        Expr output = VisitExpr(GetRef<Expr>(node->node_ref_));
        VLOG(2) << "index " << index << " added as output:\n"
                << PrettyPrint(output) << "\nat " << outputs_.size();
        expr_to_output_index_.emplace(node->node_ref_, outputs_.size());
        outputs_.emplace_back(std::move(output));
        output_types_.emplace_back(node->node_ref_->checked_type());
      }
    }
    ICHECK(!outputs_.empty());

    // Reverse the outputs so as to preserve the original evaluation order.
    std::reverse(outputs_.begin(), outputs_.end());
    std::reverse(output_types_.begin(), output_types_.end());
    num_outputs_ = outputs_.size();
    for (auto& kv : expr_to_output_index_) {
      kv.second = static_cast<int>(num_outputs_) - 1 - kv.second;
    }

    // Build the function body and type.
    Expr body;
    Type ret_type;
    if (num_outputs_ > 1) {
      body = Tuple(outputs_);
      ret_type = TupleType(output_types_);
    } else {
      body = std::move(outputs_.front());
      ret_type = std::move(output_types_.front());
    }

    // Re-express all the sub-sub-graphs in terms of the new function body.
    std::unique_ptr<DataflowGraph> new_dataflow_graph = CreateIndexedGraph(body);
    LabelledSubGraphMap pending_sub_sub_graphs;
    IndexSubst subst = MakeIndexSubst(*new_dataflow_graph);
    for (const auto& kv : sub_graph_->sub_sub_graphs_) {
      pending_sub_sub_graphs.Set(kv.first,
                                 Downcast<SubGraph>(kv.second).Subst(*new_dataflow_graph, subst));
    }

    // Rewrite to account for each sub-sub-graph, updating remaining sub-sub-graphs as we go.
    while (!pending_sub_sub_graphs.empty()) {
      auto itr = pending_sub_sub_graphs.begin();
      Extractor sub_extractor(new_dataflow_graph.get(), (*itr).second.as<SubGraphNode>(),
                              /*label=*/(*itr).first, /*function_override=*/{});
      sub_extractor.Extract();
      Rewriter rewriter(&sub_extractor);
      body = rewriter.VisitExpr(body);
      pending_sub_sub_graphs.erase((*itr).first);

      // Update dataflow graph and subst.
      std::unique_ptr<DataflowGraph> next_new_dataflow_graph = CreateIndexedGraph(body);
      subst = rewriter.MakeIndexSubst(*next_new_dataflow_graph);  // rewriter holds dataflow graph!
      new_dataflow_graph = std::move(next_new_dataflow_graph);
      LabelledSubGraphMap new_pending_sub_sub_graphs;
      // Re-express remaining sub-sub-graphs in terms of the freshly rewritten body.
      for (auto& kv : pending_sub_sub_graphs) {
        new_pending_sub_sub_graphs.Set(
            kv.first, Downcast<SubGraph>(kv.second).Subst(*new_dataflow_graph, subst));
      }
      pending_sub_sub_graphs = std::move(new_pending_sub_sub_graphs);
    }

    // Construct the extracted function.
    function_ =
        Function(std::move(params_), std::move(body), std::move(ret_type), /*ty_params=*/{});
    if (!label_.empty()) {
      function_ = WithAttr(std::move(function_), attr::kComposite, label_);
    }

    // Construct the call to either the overridden or extracted function.
    call_ = Call(function_override_.defined() ? function_override_ : function_, std::move(args_));

    // Setup the output substitution.
    for (const auto& kv : expr_to_output_index_) {
      Expr expr = num_outputs_ > 1 ? static_cast<Expr>(TupleGetItem(call_, kv.second))
                                   : static_cast<Expr>(call_);
      VLOG(2) << "output " << dataflow_graph_->item_to_node(kv.first)->index_ << " is at index "
              << kv.second << " (of " << num_outputs_ << " outputs)";
      output_substitution_.emplace(kv.first, std::move(expr));
    }

    // Cleanup accumulators
    params_.clear();
    args_.clear();
    expr_to_param_.clear();
    outputs_.clear();
    output_types_.clear();
    expr_to_output_index_.clear();
  }

  ////// Following members are valid only after Extract() has returned.

  /*!
   * \brief Returns number of outputs from the function. If greater than one then the
   * function is returning a tuple, and the \p expr_to_output_index map indicates
   * which index.
   */
  size_t num_outputs() const { return num_outputs_ > 1; }

  /*! \brief Returns the (unique) function representing the sub-graph. */
  const Function& function() const { return function_; }

  /*! \brief Returns the (unique) call to the above function to replace the sub-graph. */
  const Call& call() const { return call_; }

  /*!
   * \brief Returns the substitution to apply to all expression nodes in the overall expression
   * so as to replace references to nodes inside this sub-graph with the (tuple projection of)
   * the result of calling the above function.
   */
  const std::unordered_map<const ExprNode*, Expr>& output_substitution() const {
    return output_substitution_;
  }

 private:
  /*! \brief Returns a map from original index to new index for each node inside the sub-graph. */
  IndexSubst MakeIndexSubst(const DataflowGraph& new_dataflow_graph) const {
    VLOG(2) << "building extractor substitution";
    IndexSubst subst;
    for (PostDfsIndex index : sub_graph_->inside_) {
      auto orig_node = dataflow_graph_->index_to_node(index);
      ICHECK_EQ(orig_node->index_, index);
      auto itr = memo_.find(orig_node->ref());
      ICHECK(itr != memo_.end());
      auto new_node = new_dataflow_graph.item_to_node(itr->second);
      VLOG(2) << orig_node->index_ << " |-> " << new_node->index_;
      subst.emplace(orig_node->index_, new_node->index_);
    }
    return subst;
  }

  /*! \brief Returns true if \p expr is inside the sub-graph. */
  bool inside(const Expr& expr) {
    return sub_graph_->inside_[dataflow_graph_->item_to_node(expr)->index_];
  }

  /*!
   * \brief Returns the variable uniquely representing \p expr, which should be
   * an input node (ie outside the sub-graph but feeding into a node inside the sub-graph).
   *
   * It is valid for:
   *  - An expression outside the sub-graph to be used multiple times inside the sub-graph.
   *  - An expression outside the sub-graph to be used both inside and outside the sub-graph.
   */
  Var VarFor(const Expr& expr) {
    ICHECK(!inside(expr));
    auto itr = expr_to_param_.find(expr.get());
    if (itr != expr_to_param_.end()) {
      return itr->second;
    }
    // Ok if checked type is null here.
    auto fresh_var = Var("FunctionVar_" + std::to_string(params_.size()), expr->checked_type_);
    params_.push_back(fresh_var);
    args_.push_back(expr);
    expr_to_param_.emplace(expr.get(), fresh_var);
    return fresh_var;
  }

  /*!
   * \brief If \p expr is inside the sub-graph then return it's rewritten form.
   * Otherwise return the variable to represent it. Should be called only on inputs
   * to nodes which are inside the sub-graph.
   */
  Expr VisitExpr(const Expr& expr) final {
    if (inside(expr)) {
      return ExprMutator::VisitExpr(expr);
    } else if (CanInline(expr)) {
      // Implicitly include inlinable input sub-expressions.
      return expr;
    } else {
      return VarFor(expr);
    }
  }

  Expr VisitExpr_(const FunctionNode* function_node) override {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(function_node);
    }
    return ExprMutator::VisitExpr_(function_node);
  }

  //// Context fields, passed in constructor.

  /*! \brief The dataflow graph corresponding to the overall expression. */
  const DataflowGraph* dataflow_graph_;
  /*! \brief The sub-graph of the above we are extracting. */
  const SubGraphNode* sub_graph_;
  /*!
   * \brief If non-empty, the value of the "Composite" attribute to attach to the extracted
   * function.
   */
  String label_;
  /*! \brief If defined, the function to use instead of function_ below. */
  Function function_override_;

  //// Result fields, available after Extract() called.

  /*!
   * \brief The extracted function.
   */
  Function function_;
  /*!
   * \brief Call to either the overriding or extracted function which replaces the sub-graph.
   */
  Call call_;
  /*!
   * \brief Map from output nodes to the (tuple projection of) the above call.
   */
  std::unordered_map<const ExprNode*, Expr> output_substitution_;
  /*!
   * Number of outputs from the function body.
   */
  size_t num_outputs_ = 0;

  //// Accumulator fields, built as we visit expressions and cleared once results generated.

  /*! \brief Parameters to include in the extracted function to represent outside nodes at least
   * one inside node depends on. Ends up in the function. */
  Array<Var> params_;
  /*!
   * \brief Arguments to the extracted function corresponding to the original outside node needed
   * as an input by at least one inside node. Ends up in the call.
   */
  Array<Expr> args_;
  /*!
   * \brief Map from outside expression nodes to the parameters which should replace them in the
   * function body.
   */
  std::unordered_map<const ExprNode*, Var> expr_to_param_;
  /*!
   * \brief Accumulated new expressions which must be outputs of the function body. It is possible
   * to havel multiple outputs. It is possible one output also contributes to other outputs (ie
   * the output is a 'tap'). Ends up in the function.
   */
  std::vector<Expr> outputs_;
  std::vector<Type> output_types_;
  /*!
   * \brief Map from original outside expression nodes to the index in outputs_ which represent
   * them. This corresponds to the tuple projection index to apply to the function result to
   * recover the original expression. Ends up being used to construct the output substitution.
   */
  std::unordered_map<const ExprNode*, int> expr_to_output_index_;
};

Expr Rewriter::VisitExpr(const Expr& expr) {
  auto itr = extractor_->output_substitution().find(expr.get());
  if (itr == extractor_->output_substitution().end()) {
    return ExprMutator::VisitExpr(expr);
  } else {
    return itr->second;
  }
}

IndexSubst Rewriter::MakeIndexSubst(const DataflowGraph& new_dataflow_graph) const {
  IndexSubst subst;
  VLOG(2) << "building rewriter substitution";
  for (PostDfsIndex index = 0; index < extractor_->dataflow_graph().size(); ++index) {
    auto orig_node = extractor_->dataflow_graph().index_to_node(index);
    auto itr = memo_.find(orig_node->ref());
    if (itr != memo_.end()) {
      auto new_node = new_dataflow_graph.item_to_node(itr->second);
      VLOG(2) << orig_node->index_ << " |-> " << new_node->index_;
      subst.emplace(orig_node->index_, new_node->index_);
    }
  }
  return subst;
}

}  // namespace

std::string SubGraphConfig::ToString() const {
  std::ostringstream os;
  os << "{max_outputs=" << max_outputs;
  os << ",allow_taps=" << allow_taps;
  os << ",max_max_depth=" << max_max_depth;
  os << "}";
  return os.str();
}

std::pair<OpPatternKind, std::string> SubExprKindAndLabel(const Expr& sub_expr) {
  class Visitor : public ExprFunctor<std::pair<OpPatternKind, std::string>(const Expr&)> {
   private:
    std::pair<OpPatternKind, std::string> VisitExpr_(const CallNode* call_node) final {
      if (const auto* op_node = call_node->op.as<OpNode>()) {
        auto op = GetRef<Op>(op_node);
        static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
        if (fpattern.count(op) == 0) {
          VLOG(1) << "no TOpPattern known for " << op->name << ", considering opaque";
          return {kOpaque, op->name};
        } else if (IsDynamic(call_node->checked_type()) && IsDataDependent(call_node)) {
          VLOG(1) << "call has dynamic shape which is data-dependent, considering opaque";
          return {kOpaque, op->name};
        } else {
          OpPatternKind kind = static_cast<OpPatternKind>(fpattern[op]);
          VLOG(1) << "TOpPattern for " << op->name << " is " << KindToString(kind);
          return {kind, op->name};
        }
      } else if (const auto* function_node = call_node->op.as<FunctionNode>()) {
        Optional<Integer> opt_i =
            function_node->GetAttr<Integer>("TOpPattern", Optional<Integer>());
        if (opt_i.defined()) {
          OpPatternKind kind = static_cast<OpPatternKind>(opt_i.value()->value);
          VLOG(1) << "TOpPattern for function is " << KindToString(kind);
          return {kind, "call_prim"};
        } else {
          VLOG(1) << "calling function without TOpPattern, considering opaque";
          return {kOpaque, "call_fun"};
        }
      } else {
        VLOG(1) << "unsupported call, considering opaque";
        return {kOpaque, "call_any"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(const ConstantNode* constant_node) final {
      VLOG(1) << "TOpPattern for constant is " << KindToString(kElemWise);
      if (IsSimpleScalar(constant_node)) {
        return {kElemWise, "scalar"};
      } else {
        return {kElemWise, "const"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(const TupleNode* tuple_node) final {
      const auto* tuple_type_node = tuple_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type_node != nullptr);
      if (std::all_of(tuple_type_node->fields.begin(), tuple_type_node->fields.end(),
                      [](const Type& type) { return type.as<TensorTypeNode>() != nullptr; })) {
        VLOG(1) << "TOpPattern for tuple is " << KindToString(kInjective);
        return {kInjective, "tuple"};
      } else {
        VLOG(1) << "tuple contains non-tensors, considering opaque";
        return {kOpaque, "tuple"};
      }
    }

    std::pair<OpPatternKind, std::string> VisitExpr_(
        const TupleGetItemNode* tuple_get_item_node) final {
      const auto* tuple_type_node = tuple_get_item_node->tuple->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type_node != nullptr);
      if (std::all_of(tuple_type_node->fields.begin(), tuple_type_node->fields.end(),
                      [](const Type& type) { return type.as<TensorTypeNode>() != nullptr; })) {
        VLOG(1) << "TOpPattern for tuple projection is " << KindToString(kInjective);
        return {kInjective, "proj"};
      } else {
        VLOG(1) << "tuple being projected contains non-tensors, considering opaque";
        return {kOpaque, "proj"};
      }
    }

    // TODO(mbs): We implement the following mostly so we have a lightweight way of describing
    // the current sub-expression. If fusion is even extended beyond the usual call/tuple/proj
    // sub-language we should revise the returned operator kinds to match.

    std::pair<OpPatternKind, std::string> VisitExpr_(const VarNode* var_node) final {
      return {kOpaque, "%" + var_node->name_hint()};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const GlobalVarNode* global_var_node) final {
      return {kOpaque, "@" + global_var_node->name_hint};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const OpNode* op_node) final {
      return {kOpaque, "`" + op_node->name};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const FunctionNode* function_node) final {
      return {kOpaque, "fn"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const LetNode* let_node) final {
      return {kOpaque, "let"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const IfNode* if_node) final {
      return {kOpaque, "if"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefCreateNode* ref_create_node) final {
      return {kOpaque, "ref"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefReadNode* op) final {
      return {kOpaque, "ref_read"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const RefWriteNode* op) final {
      return {kOpaque, "ref_write"};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const ConstructorNode* op) final {
      return {kOpaque, "`" + op->name_hint};
    }
    std::pair<OpPatternKind, std::string> VisitExpr_(const MatchNode* op) final {
      return {kOpaque, "match"};
    }
  };
  return Visitor().VisitExpr(sub_expr);
}

std::pair<OpPatternKind, std::string> SubGraphKindAndLabel(const DataflowGraph& dataflow_graph,
                                                           const IndexSet& inside) {
  std::ostringstream os;
  bool first = true;
  OpPatternKind max_kind = kElemWise;
  for (PostDfsIndex index : inside) {
    OpPatternKind sub_kind;
    std::string sub_label;
    std::tie(sub_kind, sub_label) = SubExprKindAndLabel(dataflow_graph.index_to_node(index)->ref());
    if (!sub_label.empty()) {
      if (first) {
        first = false;
      } else {
        os << "+";
      }
      os << sub_label;
    }
    max_kind = CombineKinds(max_kind, sub_kind);
  }
  return {max_kind, os.str()};
}

IndexSet MatcherToIndexSet(const DFPatternMatcher& matcher) {
  IndexSet result(matcher.size());
  for (const auto& kv : matcher.memo()) {
    for (const auto& matched_sub_expr : kv.second) {
      if (CanInline(matched_sub_expr)) {
        // Trivial sub-expressions can just be included in the extracted function body
        // when we construct it and don't need to be considered part of the sub-graph.
        continue;
      }
      if (kv.first.as<WildcardPatternNode>()) {
        // Don't consider the expressions matched by a wildcard to be part of the sub-graph.
        continue;
      }
      result.Add(matcher.expr_to_node(matched_sub_expr)->index_);
    }
  }
  return result;
}

bool SubGraphNode::IsValid(const DataflowGraph& dataflow_graph,
                           const SubGraphConfig& config) const {
  // Check we don't have too many outputs.
  if (config.max_outputs > 0 && exit_.PopCount() > config.max_outputs) {
    VLOG(1) << "Subgraph " << ToString() << " is invalid: " << exit_.PopCount()
            << " outputs exceeds maximum " << config.max_outputs;
    return false;
  }

  // Check the maximum path depth is in limit.
  if (config.max_max_depth > 0 && max_depth_ > config.max_max_depth) {
    VLOG(1) << "Subgraph " << ToString() << " is invalid: maximum depth " << max_depth_
            << " exceeds limit " << config.max_max_depth;
    return false;
  }

  // All inside nodes must be in the same basic block.
  const DataflowGraph::Node* basic_block = nullptr;
  for (PostDfsIndex index : inside_) {
    auto node = dataflow_graph.index_to_node(index);
    if (basic_block == nullptr) {
      basic_block = node->basic_block_;
    }
    if (node->basic_block_ != basic_block) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: nodes are from different basic blocks";
      return false;
    }
  }

  // Any sub-sub-graphs must be subsets and non-overlapping.
  IndexSet union_bitvec(dataflow_graph.size());
  for (const auto& kv : sub_sub_graphs_) {
    if (!Downcast<SubGraph>(kv.second)->inside_.AreDisjoint(union_bitvec)) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: sub-sub-graphs overlap";
      return false;
    }
    if (!Downcast<SubGraph>(kv.second)->inside_.IsSubset(inside_)) {
      VLOG(1) << "Subgraph " << ToString()
              << " is invalid: sub-sub-graph is not subset of overall sub-graph";
      return false;
    }
  }

  if (!config.allow_taps) {
    // Exit nodes cannot also contribute to inside nodes.
    for (PostDfsIndex index : exit_) {
      auto node = dataflow_graph.index_to_node(index);
      if (AnyOutputInside(node)) {
        VLOG(1) << "Subgraph " << ToString()
                << " is invalid: inner node is 'tapped' and also contributes to output, but taps "
                   "are disabled";
        return false;
      }
    }
  }

  // Collect all nodes downstream of exit nodes.
  std::unordered_set<const DataflowGraph::Node*> downstream_nodes;
  for (PostDfsIndex index : exit_) {
    auto node = dataflow_graph.index_to_node(index);
    // Capture all downstream nodes of this output.
    node->AccumulateDownstreamNodes(downstream_nodes);
  }

  // Check no output would end up feeding into any entry node.
  for (PostDfsIndex index : entry_) {
    auto node = dataflow_graph.index_to_node(index);
    if (downstream_nodes.count(node)) {
      VLOG(1) << "Subgraph " << ToString() << " is invalid: an output node feeds into input node "
              << index;
      return false;
    }
  }

  // Looks legit!
  return true;
}

Function SubGraphNode::ExtractFunction(const DataflowGraph& dataflow_graph) const {
  Extractor extractor(&dataflow_graph, this, /*label=*/{}, /*function_override=*/{});
  extractor.Extract();
  return extractor.function();
}

Expr SubGraphNode::Partition(const DataflowGraph& dataflow_graph, const Expr& expr,
                             const Function& function_override) const {
  // Even if we are given the override function (because we have already invoked ExtractFunction
  // and possibly rewritten the result), we must still run the extractor so as to accumulate the
  // output substitution. Technically we could avoid the effort of constructing the function
  // body again but doesn't seem worth it.
  Extractor extractor(&dataflow_graph, this, /*label=*/{}, function_override);
  extractor.Extract();
  Rewriter rewriter(&extractor);
  return rewriter.VisitExpr(expr);
}

std::string SubGraphNode::ToString() const {
  std::ostringstream os;
  os << "{inside=" << inside_.ToString();
  os << ",entry=" << entry_.ToString();
  os << ",exit=" << exit_.ToString();
  os << ",input=" << input_.ToString();
  os << ",output=" << output_.ToString();
  os << ",max_depth=" << max_depth_;
  os << ",kind=" << KindToString(kind_);
  if (!label_.empty()) {
    os << ",label=" << label_;
  }
  for (const auto& kv : sub_sub_graphs_) {
    os << ",";
    os << kv.first;
    os << ":";
    os << Downcast<SubGraph>(kv.second)->ToString();
  }
  os << "}";
  return os.str();
}

bool SubGraphNode::operator==(const SubGraphNode& that) const {
  ICHECK_EQ(inside_.end_index(), that.inside_.end_index());
  if (inside_ != that.inside_) {
    return false;
  }
  if (sub_sub_graphs_.size() != that.sub_sub_graphs_.size()) {
    return false;
  }
  for (const auto kv : sub_sub_graphs_) {
    auto itr = that.sub_sub_graphs_.find(kv.first);
    if (itr == that.sub_sub_graphs_.end()) {
      return false;
    }
    if (*Downcast<SubGraph>(kv.second).get() != *Downcast<SubGraph>((*itr).second).get()) {
      return false;
    }
  }
  return true;
}

size_t SubGraphNode::hash() const {
  size_t h = inside_.hash();
  std::vector<std::string> labels;
  for (const auto kv : sub_sub_graphs_) {
    labels.push_back(kv.first);
  }
  std::sort(labels.begin(), labels.end());
  std::hash<std::string> str_hasher;
  for (const auto& label : labels) {
    h ^= str_hasher(label) + 0x9e3779b9 + (h << 6) + (h >> 2);
    auto itr = sub_sub_graphs_.find(label);
    ICHECK(itr != sub_sub_graphs_.end());
    h ^= Downcast<SubGraph>((*itr).second)->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  return h;
}

void SubGraphNode::Init(const DataflowGraph& dataflow_graph) {
  for (PostDfsIndex index = 0; index < inside_.end_index(); ++index) {
    auto node = dataflow_graph.index_to_node(index);
    if (inside_[index]) {
      if (AnyInputOutside(node)) {
        entry_.Add(index);
      }
      if (AnyOutputOutside(node) || node->is_external_) {
        exit_.Add(index);
      }
    } else {
      if (AnyInputInside(node)) {
        output_.Add(index);
      }
      if (AnyOutputInside(node) && !CanInline(node->ref())) {
        input_.Add(index);
      }
    }
  }
  max_depth_ = MaxDepth(dataflow_graph);
}

size_t SubGraphNode::MaxDepth(const DataflowGraph& dataflow_graph) const {
  std::unordered_map<const DataflowGraph::Node*, size_t> max_depths;
  std::vector<const DataflowGraph::Node*> stack;
  size_t max_depth = 0;
  // All the entry nodes have max depth 0.
  for (PostDfsIndex index : entry_) {
    auto node = dataflow_graph.index_to_node(index);
    max_depths.emplace(node, 0);
    stack.push_back(node);
  }
  while (!stack.empty()) {
    const DataflowGraph::Node* node = stack.back();
    stack.pop_back();
    size_t next_depth = max_depths[node] + 1;
    if (exit_[node->index_]) {
      // If this node is external then it will have no outputs but we still wish to consider
      // the path to the implied output as requiring one more step.
      // Otherwise we're accounting for reaching one of the external outputs belowe.
      max_depth = std::max(max_depth, next_depth);
    }
    for (const DataflowGraph::Node* output_node : node->outputs_) {
      if (!inside_[output_node->index_]) {
        continue;
      }
      if (max_depths.count(output_node) == 0) {
        max_depths.emplace(output_node, next_depth);
        stack.push_back(output_node);
      } else if (next_depth > max_depths[output_node]) {
        // We found a deeper path to an already expanded node. We'll expand again.
        max_depths[output_node] = next_depth;
        stack.push_back(output_node);
      }
    }
  }
  return max_depth;
}

/*! \brief Return's true if any (input/output) of node is (outside/inside) the sub-graph.  */
bool SubGraphNode::AnyInputOutside(const DataflowGraph::Node* node) const {
  return std::any_of(node->inputs_.begin(), node->inputs_.end(),
                     [this](const DataflowGraph::Node* sub_node) {
                       return !inside_[sub_node->index_] && !CanInline(sub_node->ref());
                     });
}

bool SubGraphNode::AnyInputInside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->inputs_.begin(), node->inputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return inside_[sub_node->index_]; });
}

bool SubGraphNode::AnyOutputOutside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->outputs_.begin(), node->outputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return !inside_[sub_node->index_]; });
}

bool SubGraphNode::AnyOutputInside(const DataflowGraph::Node* node) const {
  return std::any_of(
      node->outputs_.begin(), node->outputs_.end(),
      [this](const DataflowGraph::Node* sub_node) { return inside_[sub_node->index_]; });
}

SubGraph::SubGraph(const DataflowGraph& dataflow_graph, IndexSet inside, OpPatternKind kind,
                   String label, LabelledSubGraphMap sub_sub_graphs) {
  auto node = runtime::make_object<SubGraphNode>();
  node->inside_ = std::move(inside);
  node->first_inside_index_ = node->inside_.FirstInsideIndex();
  node->last_inside_index_ = node->inside_.LastInsideIndex();
  node->entry_ = IndexSet(node->inside_.end_index());
  node->exit_ = IndexSet(node->inside_.end_index());
  node->input_ = IndexSet(node->inside_.end_index());
  node->output_ = IndexSet(node->inside_.end_index());
  node->kind_ = kind;
  node->label_ = std::move(label);
  node->sub_sub_graphs_ = std::move(sub_sub_graphs);
  node->Init(dataflow_graph);
  data_ = std::move(node);
}

SubGraph::SubGraph(const DataflowGraph& dataflow_graph)
    : SubGraph(dataflow_graph, IndexSet(dataflow_graph.size())) {}

bool SubGraph::AreDisjoint(const SubGraph& that) const {
  if (!get()->inside_.AreDisjoint(that->inside_)) {
    return false;
  }
  for (const auto& kv : get()->sub_sub_graphs_) {
    if (that->sub_sub_graphs_.count(kv.first)) {
      return false;
    }
  }
  return true;
}

SubGraph SubGraph::DisjointUnion(const DataflowGraph& dataflow_graph, const SubGraph& that) const {
  ICHECK(AreDisjoint(that));
  IndexSet inside = get()->inside_ | that->inside_;
  LabelledSubGraphMap sub_sub_graphs;
  for (const auto& kv : get()->sub_sub_graphs_) {
    sub_sub_graphs.Set(kv.first, kv.second);
  }
  for (const auto& kv : that->sub_sub_graphs_) {
    sub_sub_graphs.Set(kv.first, kv.second);
  }
  return SubGraph(dataflow_graph, std::move(inside), CombineKinds(get()->kind_, that->kind_),
                  UnionLabels(get()->label_, that->label_), std::move(sub_sub_graphs));
}

SubGraph SubGraph::WithLabel(const DataflowGraph& dataflow_graph, String label) const {
  LabelledSubGraphMap sub_sub_graphs;
  sub_sub_graphs.Set(std::move(label), *this);
  return SubGraph(dataflow_graph, get()->inside_, get()->kind_, get()->label_,
                  std::move(sub_sub_graphs));
}

SubGraph SubGraph::Subst(const DataflowGraph& new_dataflow_graph, const IndexSubst& subst) const {
  IndexSet new_inside = get()->inside_.Subst(new_dataflow_graph.size(), subst);
  LabelledSubGraphMap new_sub_sub_graphs;
  for (const auto& kv : get()->sub_sub_graphs_) {
    new_sub_sub_graphs.Set(kv.first,
                           Downcast<SubGraph>(kv.second).Subst(new_dataflow_graph, subst));
  }
  return SubGraph(new_dataflow_graph, std::move(new_inside), get()->kind_, get()->label_,
                  std::move(new_sub_sub_graphs));
}

transform::Pass PartitionOnIndexesForTesting(size_t max_outputs, bool allow_taps,
                                             Array<Integer> indexes, Array<String> labels) {
  auto pass_func = [=](Function function, IRModule mod, transform::PassContext ctxt) {
    ICHECK(!labels.defined() || indexes.size() == labels.size());
    VLOG(1) << "Considering partitioning for:\n" << PrettyPrint(function);
    std::unique_ptr<DataflowGraph> dataflow_graph = CreateIndexedGraph(function);
    std::unordered_map<String, std::vector<PostDfsIndex>> sub_sub_graph_indexes;
    std::vector<PostDfsIndex> node_indexes;
    node_indexes.reserve(indexes.size());
    for (size_t i = 0; i < indexes.size(); ++i) {
      const Integer& index = indexes[i];
      ICHECK_GE(index->value, 0);
      ICHECK_LT(index->value, dataflow_graph->size());
      PostDfsIndex index_int = static_cast<PostDfsIndex>(index->value);
      node_indexes.push_back(index_int);
      if (labels.defined()) {
        const String& label = labels[i];
        if (!label.empty()) {
          sub_sub_graph_indexes[label].push_back(index_int);
        }
      }
    }
    LabelledSubGraphMap sub_sub_graphs;
    for (const auto& kv : sub_sub_graph_indexes) {
      sub_sub_graphs.Set(kv.first,
                         SubGraph(*dataflow_graph, IndexSet(dataflow_graph->size(), kv.second)));
    }
    OpPatternKind kind;
    String label;
    IndexSet inside(dataflow_graph->size(), node_indexes);
    std::tie(kind, label) = SubGraphKindAndLabel(*dataflow_graph, inside);
    SubGraph sub_graph(*dataflow_graph, std::move(inside), kind, std::move(label),
                       std::move(sub_sub_graphs));
    SubGraphConfig config;
    config.max_outputs = max_outputs;
    config.allow_taps = allow_taps;
    if (sub_graph->IsValid(*dataflow_graph, config)) {
      VLOG(1) << "Sub-graph " << sub_graph->ToString() << " is considered valid";
    } else {
      VLOG(1) << "Sub-graph " << sub_graph->ToString()
              << " is NOT considered valid, not partitioning";
      return function;
    }
    Function partitioned_function = sub_graph->ExtractFunction(*dataflow_graph);
    Function result =
        Downcast<Function>(sub_graph->Partition(*dataflow_graph, function, partitioned_function));
    VLOG(1) << "Partitioned to:\n" << PrettyPrint(result);
    return result;
  };
  return transform::CreateFunctionPass(pass_func, /*opt_level=*/0, "PartitionOnIndexesForTesting",
                                       {});
}

TVM_REGISTER_GLOBAL("relay.collage.partition_on_indexes_for_testing")
    .set_body_typed([](size_t max_outputs, bool allow_taps, Array<Integer> indexes,
                       Array<String> labels) {
      return PartitionOnIndexesForTesting(max_outputs, allow_taps, indexes, labels);
    });

}  // namespace collage
}  // namespace relay
}  // namespace tvm