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
 * \file tvm/relay/collage/capture_index_in_spans.cc
 * \brief Pass to set spans to capture the post-dfs index of every node. For debuggin only.
 */

#include "./capture_index_in_spans.h"

#include <tvm/relay/expr_functor.h>

#include "../ir/indexed_graph.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {
class CaptureIndexInSpansRewriter : public ExprMutator {
 public:
  explicit CaptureIndexInSpansRewriter(const DataflowGraph* dataflow_graph)
      : source_name_(SourceName::Get("index")), dataflow_graph_(dataflow_graph) {}

 private:
  Expr VisitExpr_(const VarNode* var_node) override {
    auto var = GetRef<Var>(var_node);
    MakeSpan(var);  // for side effects
    return var;
  }

  Expr VisitExpr_(const ConstantNode* constant_node) override {
    auto constant = GetRef<Constant>(constant_node);
    return WithFields(constant, {}, {}, MakeSpan(constant));
  }

  Expr VisitExpr_(const GlobalVarNode* global_var_node) override {
    auto global_var = GetRef<GlobalVar>(global_var_node);
    MakeSpan(global_var);  // for side effects
    return global_var;
  }

  Expr VisitExpr_(const OpNode* op_node) override {
    auto op = GetRef<Op>(op_node);
    MakeSpan(op);  // for side effects
    return op;
  }

  Expr VisitExpr_(const TupleNode* tuple_node) override {
    auto tuple = GetRef<Tuple>(tuple_node);
    auto new_tuple = Downcast<Tuple>(ExprMutator::VisitExpr_(tuple_node));
    return WithFields(new_tuple, {}, {}, MakeSpan(tuple));
  }

  Expr VisitExpr_(const FunctionNode* function_node) override {
    auto function = GetRef<Function>(function_node);
    // Don't recurse into the bodies of primitive functions.
    // CAUTION: This is why we can't just use an ExprRewriter.
    Function new_function = function_node->HasNonzeroAttr(attr::kPrimitive)
                                ? function
                                : Downcast<Function>(ExprMutator::VisitExpr_(function_node));
    return WithFields(new_function, {}, {}, {}, {}, {}, {}, MakeSpan(function));
  }

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = GetRef<Call>(call_node);
    auto new_call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    return WithFields(new_call, {}, {}, {}, {}, {}, MakeSpan(call));
  }

  Expr VisitExpr_(const LetNode* let_node) override {
    auto let = GetRef<Let>(let_node);
    auto new_let = Downcast<Let>(ExprMutator::VisitExpr_(let_node));
    return WithFields(new_let, {}, {}, {}, {}, MakeSpan(let));
  }

  Expr VisitExpr_(const IfNode* if_node) override {
    auto ife = GetRef<If>(if_node);
    auto new_ife = Downcast<If>(ExprMutator::VisitExpr_(if_node));
    return WithFields(new_ife, {}, {}, {}, {}, MakeSpan(ife));
  }

  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item_node) override {
    auto tuple_get_item = GetRef<TupleGetItem>(tuple_get_item_node);
    auto new_tuple_get_item = Downcast<TupleGetItem>(ExprMutator::VisitExpr_(tuple_get_item_node));
    return WithFields(new_tuple_get_item, {}, {}, {}, MakeSpan(tuple_get_item));
  }

  Expr VisitExpr_(const RefCreateNode* ref_create_node) override {
    auto ref_create = GetRef<RefCreate>(ref_create_node);
    auto new_ref_create = Downcast<RefCreate>(ExprMutator::VisitExpr_(ref_create_node));
    return WithFields(new_ref_create, {}, {}, MakeSpan(ref_create));
  }

  Expr VisitExpr_(const RefReadNode* ref_read_node) override {
    auto ref_read = GetRef<RefRead>(ref_read_node);
    auto new_ref_read = Downcast<RefRead>(ExprMutator::VisitExpr_(ref_read_node));
    return WithFields(new_ref_read, {}, {}, MakeSpan(ref_read));
  }

  Expr VisitExpr_(const RefWriteNode* ref_write_node) override {
    auto ref_write = GetRef<RefWrite>(ref_write_node);
    auto new_ref_write = Downcast<RefWrite>(ExprMutator::VisitExpr_(ref_write_node));
    return WithFields(new_ref_write, {}, {}, {}, MakeSpan(ref_write));
  }

  Expr VisitExpr_(const ConstructorNode* constructor_node) override {
    auto constructor = GetRef<Constructor>(constructor_node);
    MakeSpan(constructor);  // for side effects
    return constructor;
  }

  Expr VisitExpr_(const MatchNode* match_node) override {
    auto match = GetRef<Match>(match_node);
    auto new_match = Downcast<Match>(ExprMutator::VisitExpr_(match_node));
    return WithFields(new_match, {}, {}, {}, MakeSpan(match));
  }

  Span MakeSpan(const Expr& expr) {
    auto node = dataflow_graph_->item_to_node(expr);
    PostDfsIndex node_index = node->index_;
    PostDfsIndex dominator_index = node->dominator_parent_ ? node->dominator_parent_->index_ : -1;
    Span span(source_name_, /*line=*/node_index, /*end_line=*/node_index,
              /*column=*/dominator_index, /*end_column=*/dominator_index);
    ICHECK_EQ(index_, node_index)
        << "expecting visit order to match dataflow graph's post-dfs index order at expression:\n"
        << PrettyPrint(expr);
    index_++;
    return span;
  }

  SourceName source_name_;
  const DataflowGraph* dataflow_graph_;
  PostDfsIndex index_ = 0;
};

}  // namespace

/*!
 * Captures the post-dfs index and dominator post-dfs index of every node in it's span, in the form
 *   "index:<post-dfs index>:<dominator post-dfs index>
 * For debugging only.
 */
transform::Pass CaptureIndexInSpans() {
  auto pass_func = [](Function f, IRModule m, transform::PassContext ctxt) {
    std::unique_ptr<DataflowGraph> dataflow_graph = CreateIndexedGraph(f);
    CaptureIndexInSpansRewriter rewriter(dataflow_graph.get());
    return Downcast<Function>(rewriter.VisitExpr(f));
  };
  return transform::CreateFunctionPass(pass_func, 0, "CaptureIndexInSpans", {});
};

TVM_REGISTER_GLOBAL("relay.collage.capture_index_in_spans").set_body_typed(CaptureIndexInSpans);

}  // namespace collage
}  // namespace relay
}  // namespace tvm
