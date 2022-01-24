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
 * \file src/relay/ir/indexed_graph.cc
 * \brief Utilties for Creating Indexed Graphs.
 */
#include "indexed_graph.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_pattern_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

namespace tvm {
namespace relay {

std::unique_ptr<DataflowGraph> CreateIndexedGraph(const Expr& expr) {
  /*! \brief Creates an IndexedGraph and determines topological order */
  class Creator : public MixedModeVisitor {
   public:
    std::unique_ptr<DataflowGraph> CreateGraph(const Expr& expr) {
      graph_ = std::make_unique<DataflowGraph>();
      VisitExpr(expr);
      graph_->item_to_node(expr)->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    using MixedModeVisitor::VisitExpr_;

    void VisitLeaf(const Expr& expr) override {
      MixedModeVisitor::VisitLeaf(expr);
      graph_->AddNode(expr);
    }

    void VisitExpr_(const FunctionNode* function_node) override {
      if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        // Don't recurse into primitive functions.
        return;
      }
      ExprVisitor::VisitExpr_(function_node);
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto pre_visit = [&](const LetNode* op) {
        VisitExpr(op->value);
        VisitExpr(op->var);
      };
      auto post_visit = [&](const LetNode* op) {
        VisitExpr(op->body);
        if (let_node != op) {
          visit_counter_[op]++;
          graph_->AddNode(GetRef<Expr>(op));
        }
      };
      ExpandANormalForm(let_node, pre_visit, post_visit);
    }

    class PatternCreator : public PatternVisitor {
     public:
      explicit PatternCreator(Creator* creator) : creator_(creator) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        creator_->graph_->AddNode(pattern_var_node->var);
      }

      Creator* creator_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      VisitExpr(match_node->data);
      for (const Clause& c : match_node->clauses) {
        PatternCreator pattern_creator(this);
        pattern_creator.VisitPattern(c->lhs);
        VisitExpr(c->rhs);
      }
    }

    std::unique_ptr<DataflowGraph> graph_;
  };

  /*!
   * \brief Takes an IndexedGraph, fills it's forward outputs, and does dominator tree
   * analysis.
   *
   * Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public ExprFunctor<void(const Expr&)> {
   public:
    explicit Annotator(std::unique_ptr<DataflowGraph> graph) : graph_(std::move(graph)) {}

    std::unique_ptr<DataflowGraph> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (PostDfsIndex index = 0; index < graph_->size(); ++index) {
        VisitExpr(graph_->index_to_node(index)->ref());
      }
      // do the dominator analysis
      graph_->PostDom();
      for (PostDfsIndex index = 0; index < graph_->size(); ++index) {
        auto node = graph_->index_to_node(index);
        if (node->dominator_parent_) {
          VLOG(2) << "node index " << index << " has dominator parent index "
                  << node->dominator_parent_->index_;
        }
      }
      return std::move(graph_);
    }

    /*!
     * \brief Add \p parent as a possible output of the node corresponding to \p expr.
     */
    void AddOutput(const Expr& expr, DataflowGraph::Node* parent) {
      auto current = graph_->item_to_node(expr);
      current->outputs_.push_back(parent);
      parent->inputs_.push_back(current);
    }

   protected:
    void VisitExpr_(const VarNode* var_node) override {}

    void VisitExpr_(const GlobalVarNode* global_var_node) override {}

    void VisitExpr_(const ConstantNode* constant_node) override {}

    void VisitExpr_(const TupleNode* tuple_node) override {
      auto node = graph_->item_to_node(GetRef<Tuple>(tuple_node));
      for (auto field : tuple_node->fields) {
        AddOutput(field, node);
      }
    }

    void VisitExpr_(const FunctionNode* function_node) override {
      if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        // No dataflow analysis inside primitive functions
        return;
      }
      auto node = graph_->item_to_node(GetRef<Function>(function_node));
      // Nothing to do for parameters -- each use of a parameter will contribute to its outputs.
      AddOutput(function_node->body, node);
    }

    void VisitExpr_(const CallNode* call_node) override {
      auto node = graph_->item_to_node(GetRef<Call>(call_node));
      AddOutput(call_node->op, node);
      for (auto arg : call_node->args) {
        AddOutput(arg, node);
      }
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto node = graph_->item_to_node(GetRef<Let>(let_node));
      auto let_var_node = graph_->item_to_node(let_node->var);
      AddOutput(let_node->value, let_var_node);
      // Nothing to do for the let-bound variable -- each use of that variable in the let-body
      // will contribute to its outputs.
      AddOutput(let_node->body, node);
    }

    void VisitExpr_(const IfNode* if_node) override {
      auto node = graph_->item_to_node(GetRef<If>(if_node));
      AddOutput(if_node->cond, node);
      AddOutput(if_node->true_branch, node);
      AddOutput(if_node->false_branch, node);
    }

    void VisitExpr_(const OpNode* op_node) override {}

    void VisitExpr_(const TupleGetItemNode* tuple_get_item_node) override {
      auto node = graph_->item_to_node(GetRef<TupleGetItem>(tuple_get_item_node));
      AddOutput(tuple_get_item_node->tuple, node);
    }

    void VisitExpr_(const RefCreateNode* ref_create_node) override {
      auto node = graph_->item_to_node(GetRef<RefCreate>(ref_create_node));
      AddOutput(ref_create_node->value, node);
    }

    void VisitExpr_(const RefReadNode* ref_read_node) override {
      auto node = graph_->item_to_node(GetRef<RefRead>(ref_read_node));
      AddOutput(ref_read_node->ref, node);
    }

    void VisitExpr_(const RefWriteNode* ref_write_node) override {
      auto node = graph_->item_to_node(GetRef<RefWrite>(ref_write_node));
      AddOutput(ref_write_node->ref, node);
      AddOutput(ref_write_node->value, node);
    }

    void VisitExpr_(const ConstructorNode* constructor_node) override {}

    class PatternAnnotator : public PatternVisitor {
     public:
      PatternAnnotator(Annotator* annotator, const ExprNode* adt_node)
          : annotator_(annotator), adt_node_(adt_node) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        auto node = annotator_->graph_->item_to_node(pattern_var_node->var);
        annotator_->AddOutput(GetRef<Expr>(adt_node_), node);
      }

      Annotator* annotator_;
      const ExprNode* adt_node_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      // Data flows from the match data to pattern vars into match arms and out into overall match.
      auto node = graph_->item_to_node(GetRef<Match>(match_node));
      for (const Clause& c : match_node->clauses) {
        PatternAnnotator pattern_annotator(this, match_node->data.get());
        pattern_annotator.VisitPattern(c->lhs);
        AddOutput(c->rhs, node);
      }
    }

    std::unique_ptr<DataflowGraph> graph_;
  };

  /*! \brief Fills in the basic blocks for all nodes. */
  class Blocker : public MixedModeVisitor {
   public:
    explicit Blocker(std::unique_ptr<DataflowGraph> graph) : graph_(std::move(graph)) {}

    std::unique_ptr<DataflowGraph> Scope(const Expr& expr) {
      VisitExpr(expr);
      return std::move(graph_);
    }

   private:
    using MixedModeVisitor::VisitExpr_;

    void VisitLeaf(const Expr& expr) override {
      MixedModeVisitor::VisitLeaf(expr);
      SetScope(expr);
    }

    void VisitExpr_(const FunctionNode* function_node) override {
      if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        return;
      }
      auto node = graph_->item_to_node(GetRef<Function>(function_node));
      basic_block_stack_.push_back(node);
      ExprVisitor::VisitExpr_(function_node);
      basic_block_stack_.pop_back();
    }

    void VisitExpr_(const IfNode* if_node) override {
      VisitExpr(if_node->cond);
      auto node = graph_->item_to_node(GetRef<If>(if_node));
      basic_block_stack_.push_back(node);
      VisitExpr(if_node->true_branch);
      VisitExpr(if_node->false_branch);
      basic_block_stack_.pop_back();
    }

    void VisitExpr_(const LetNode* let_node) override {
      auto pre_visit = [&](const LetNode* op) {
        VisitExpr(op->value);
        VisitExpr(op->var);
      };
      auto post_visit = [&](const LetNode* op) {
        VisitExpr(op->body);
        if (let_node != op) {
          visit_counter_[op]++;
          SetScope(GetRef<Let>(op));
        }
      };
      ExpandANormalForm(let_node, pre_visit, post_visit);
    }

    class PatternBlocker : public PatternVisitor {
     public:
      explicit PatternBlocker(Blocker* scoper) : scoper_(scoper) {}

     private:
      void VisitPattern_(const PatternVarNode* pattern_var_node) final {
        scoper_->SetScope(pattern_var_node->var);
      }

      Blocker* scoper_;
    };

    void VisitExpr_(const MatchNode* match_node) override {
      VisitExpr(match_node->data);
      auto node = graph_->item_to_node(GetRef<Match>(match_node));
      basic_block_stack_.push_back(node);
      for (const Clause& c : match_node->clauses) {
        PatternBlocker pattern_scoper(this);
        pattern_scoper.VisitPattern(c->lhs);
        VisitExpr(c->rhs);
      }
      basic_block_stack_.pop_back();
    }

    void SetScope(const Expr& expr) {
      auto node = graph_->item_to_node(expr);
      if (!basic_block_stack_.empty()) {
        node->basic_block_ = basic_block_stack_.back();
        VLOG(2) << "node index " << node->index_ << " has basic block index "
                << node->basic_block_->index_;
      } else {
        VLOG(2) << "node index " << node->index_ << " has no basic block";
      }
    }

    std::unique_ptr<DataflowGraph> graph_;
    std::vector<DataflowGraph::Node*> basic_block_stack_;
  };

  return Blocker(Annotator(Creator().CreateGraph(expr)).Annotate()).Scope(expr);
}

std::unique_ptr<PatternGraph> CreateIndexedGraph(const DFPattern& pattern) {
  /*! \brief Creates an IndexedGraph and determines topological order */
  class Creator : public DFPatternVisitor {
   public:
    std::unique_ptr<PatternGraph> CreateGraph(const DFPattern& pattern) {
      graph_ = std::make_unique<PatternGraph>();
      VisitDFPattern(pattern);
      graph_->item_to_node(pattern)->is_external_ = true;
      return std::move(graph_);
    }

   protected:
    void VisitDFPattern(const DFPattern& pattern) override {
      if (this->visited_.count(pattern.get()) == 0) {
        DFPatternVisitor::VisitDFPattern(pattern);
        graph_->AddNode(pattern);
      }
    }

    std::unique_ptr<PatternGraph> graph_;
  };

  /*! \brief Annotator takes an IndexedGraph, fills it's forward outputs, and does domiantor tree
   * analysis.
   *
   *  Annotator use ExprFunctor to visit nodes, but iterates over them in pre-determined
   * topological order instead of recursing.
   */
  class Annotator : public DFPatternFunctor<void(const DFPattern&)> {
   public:
    Annotator(std::unique_ptr<PatternGraph> graph) : graph_(std::move(graph)) {}

    std::unique_ptr<PatternGraph> Annotate() {
      // Visit all of the nodes in topological order to get forward outputs
      for (PostDfsIndex index = 0; index < graph_->size(); ++index) {
        VisitDFPattern(graph_->index_to_node(index)->ref());
      }
      // do the dominator analysis
      graph_->PostDom();
      return std::move(graph_);
    }

    /*! Default visitation pushes the parent to the child's outputs */
    void AddOutput(const DFPattern& pattern, PatternGraph::Node* parent) {
      auto current = graph_->item_to_node(pattern);
      if (parent) {
        current->outputs_.push_back(parent);
        parent->inputs_.push_back(current);
      }
    }

   protected:
    void VisitDFPattern_(const AltPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<AltPattern>(op));
      AddOutput(op->left, node);
      AddOutput(op->right, node);
    }

    void VisitDFPattern_(const AttrPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<AttrPattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const CallPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<CallPattern>(op));
      AddOutput(op->op, node);
      if (op->args.defined()) {
        for (auto arg : op->args) {
          AddOutput(arg, node);
        }
      }
    }

    void VisitDFPattern_(const ConstantPatternNode* op) override {}

    void VisitDFPattern_(const DataTypePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<DataTypePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const DominatorPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<DominatorPattern>(op));
      AddOutput(op->parent, node);
      AddOutput(op->path, node);
      AddOutput(op->child, node);
    }

    void VisitDFPattern_(const ExprPatternNode* op) override {}

    void VisitDFPattern_(const FunctionPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<FunctionPattern>(op));
      if (op->params.defined()) {
        for (auto param : op->params) {
          AddOutput(param, node);
        }
      }
      AddOutput(op->body, node);
    }

    void VisitDFPattern_(const ShapePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<ShapePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const TupleGetItemPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TupleGetItemPattern>(op));
      AddOutput(op->tuple, node);
    }

    void VisitDFPattern_(const TuplePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TuplePattern>(op));
      if (op->fields.defined()) {
        for (auto field : op->fields) {
          AddOutput(field, node);
        }
      }
    }

    void VisitDFPattern_(const IfPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<IfPattern>(op));
      AddOutput(op->cond, node);
      AddOutput(op->true_branch, node);
      AddOutput(op->false_branch, node);
    }

    void VisitDFPattern_(const LetPatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<LetPattern>(op));
      AddOutput(op->var, node);
      AddOutput(op->value, node);
      AddOutput(op->body, node);
    }

    void VisitDFPattern_(const TypePatternNode* op) override {
      auto node = graph_->item_to_node(GetRef<TypePattern>(op));
      AddOutput(op->pattern, node);
    }

    void VisitDFPattern_(const VarPatternNode* op) override {}

    void VisitDFPattern_(const WildcardPatternNode* op) override {}

    std::unique_ptr<PatternGraph> graph_;
  };

  return Annotator(Creator().CreateGraph(pattern)).Annotate();
}

}  // namespace relay
}  // namespace tvm
