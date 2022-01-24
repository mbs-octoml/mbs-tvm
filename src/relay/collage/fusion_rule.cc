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
 * \file src/relay/collage/fusion_rule.cc
 * \brief Fusion patterns and rules.
 */

#include "./fusion_rule.h"

#include <tvm/relay/transform.h>

#include "utils.h"

namespace tvm {
namespace relay {
namespace collage {

Array<CandidateKernel> FusionRuleNode::AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                           const FusionSpec& spec,
                                                           NameSupply* name_supply) const {
  ICHECK(false) << "FusionRuleNode::AllCandidateKernels should be overridden in sub-class";
  return {};
}

std::string FusionRuleNode::ToString() const { return ToDoc().str(); }

Doc FusionRuleNode::ToDoc() const {
  Doc doc;
  doc << GetTypeKey() << "(" << Doc::NewLine(2);
  std::vector<Doc> body_items;
  AppendBodyItems(body_items);
  doc << Doc::Indent(2, Doc::Concat(body_items, Doc::NewLine())) << Doc::NewLine();
  doc << ")";
  return doc;
}

void FusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  body_items.emplace_back();
  body_items.back() << "rule_name=" << Doc::StrLiteral(rule_name_);
  body_items.emplace_back();
  body_items.back() << "priority=" << priority_;
}

FusionRule::FusionRule(String rule_name, int priority) {
  auto node = runtime::make_object<FusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  data_ = std::move(node);
}

bool DefaultPatternPredicate(const Expr& matched_sub_expr) { return true; }

Array<CandidateKernel> DFPatternFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  Array<CandidateKernel> result;
  DFPatternMatcher matcher(&dataflow_graph);
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    Expr sub_expr = dataflow_graph.index_to_node(index)->ref();
    if (!matcher.Match(pattern_, sub_expr)) {
      continue;
    }
    if (!predicate_(sub_expr)) {
      continue;
    }
    IndexSet inside = MatcherToIndexSet(matcher);
    OpPatternKind kind;
    String label;
    std::tie(kind, label) = SubGraphKindAndLabel(dataflow_graph, inside);
    SubGraph sub_graph(dataflow_graph, std::move(inside), kind, std::move(label));
    String rule_name = NestLabels(rule_name_, sub_graph->label_);
    CandidateKernel candidate(std::move(rule_name), priority_, std::move(sub_graph), spec);
    VLOG(1) << "DFPatternFusionRule(" << rule_name_ << ") yields " << candidate->ToString();
    result.push_back(candidate);
  }
  return result;
}

void DFPatternFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "pattern=" << PrettyPrint(pattern_);
}

DFPatternFusionRule::DFPatternFusionRule(String rule_name, int priority, DFPattern pattern,
                                         TPatternPredicate predicate) {
  auto node = runtime::make_object<DFPatternFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->pattern_ = std::move(pattern);
  node->predicate_ = std::move(predicate);
  data_ = std::move(node);
}

Array<CandidateKernel> OpPredicateFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  Array<CandidateKernel> result;
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    auto node = dataflow_graph.index_to_node(index);
    Expr sub_expr = node->ref();
    // Is this a call to an operator?
    const auto* call_node = sub_expr.as<CallNode>();
    if (call_node == nullptr) {
      continue;
    }
    const auto* op_node = call_node->op.as<OpNode>();
    if (op_node == nullptr) {
      continue;
    }
    auto op = GetRef<Op>(op_node);
    // Does the op's predicate fire?
    if (!Op::HasAttrMap(attribute_)) {
      continue;
    }
    OpAttrMap<TPatternPredicate> fannotate = Op::GetAttrMap<TPatternPredicate>(attribute_);
    if (!fannotate.count(op)) {
      continue;
    }
    if (!fannotate[op](sub_expr)) {
      continue;
    }
    // We don't know how deep into the sub-expression the predicate looked.
    // First assume none of the arguments.
    Expr extracted_sub_expr;
    IndexSet inside;
    std::tie(extracted_sub_expr, inside) =
        Extract(dataflow_graph, node, /*include_const_args=*/false);
    if (!fannotate[op](extracted_sub_expr)) {
      // Ok, try again, but include any constant args.
      std::tie(extracted_sub_expr, inside) =
          Extract(dataflow_graph, node, /*include_const_args=*/true);
      if (fannotate[op](extracted_sub_expr)) {
        VLOG(1) << "including call and some args:\n" << PrettyPrint(extracted_sub_expr);
      } else {
        // Give up. If a predicate-based BYOC integration is looking deeply into call
        // sub-expressions it suggests it should be reworked to be pattern-based.
        VLOG(1) << "predicate returned false on extracted sub-expression:\n"
                << PrettyPrint(extracted_sub_expr);
        continue;
      }
    }
    OpPatternKind kind;
    String label;
    std::tie(kind, label) = SubExprKindAndLabel(sub_expr);
    SubGraph sub_graph(dataflow_graph, std::move(inside), kind, std::move(label));
    String rule_name = NestLabels(rule_name_, sub_graph->label_);
    CandidateKernel candidate(std::move(rule_name), priority_, std::move(sub_graph), spec);
    VLOG(1) << "OpPredicateFusionRule(" << rule_name_ << ") yields " << candidate->ToString();
    result.push_back(candidate);
  }
  return result;
}

/* static */
std::pair<Expr, IndexSet> OpPredicateFusionRuleNode::Extract(const DataflowGraph& dataflow_graph,
                                                             const DataflowGraph::Node* node,
                                                             bool include_const_args) {
  IndexSet inside(dataflow_graph.size(), {node->index_});
  if (include_const_args) {
    for (const auto* input_node : node->inputs_) {
      if (input_node->ref().as<ConstantNode>()) {
        inside.Add(input_node->index_);
      }
    }
  }
  SubGraph sub_graph(dataflow_graph, inside);
  Function function = sub_graph->ExtractFunction(dataflow_graph);
  return {InferType(function->body), inside};
}

void OpPredicateFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "attribute=" << Doc::StrLiteral(attribute_);
}

OpPredicateFusionRule::OpPredicateFusionRule(String rule_name, int priority, String attribute) {
  auto node = runtime::make_object<OpPredicateFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->attribute_ = std::move(attribute);
  data_ = std::move(node);
}

Array<CandidateKernel> CompositeFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  Array<CandidateKernel> candidates =
      sub_rule_->AllCandidateKernels(dataflow_graph, spec, name_supply);
  Array<CandidateKernel> result;
  for (auto& candidate : candidates) {
    CandidateKernel new_candidate(
        NestLabels(rule_name_, candidate->rule_name_), candidate->priority_,
        candidate->sub_graph_.WithLabel(dataflow_graph, rule_name_), candidate->spec_);
    VLOG(1) << "CompositeFusionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.push_back(new_candidate);
  }
  return result;
}

void CompositeFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
}

CompositeFusionRule::CompositeFusionRule(String rule_name, int priority, FusionRule sub_rule) {
  auto node = runtime::make_object<CompositeFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

Array<CandidateKernel> UnionFusionRuleNode::AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                                const FusionSpec& spec,
                                                                NameSupply* name_supply) const {
  Array<CandidateKernel> result;
  for (const auto& sub_rule : sub_rules_) {
    Array<CandidateKernel> candidates =
        sub_rule->AllCandidateKernels(dataflow_graph, spec, name_supply);
    for (auto& candidate : candidates) {
      CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                    candidate->priority_, candidate->sub_graph_, candidate->spec_);
      VLOG(1) << "UnionFusionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
      result.push_back(candidate);
    }
  }
  return result;
}

void UnionFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  for (const auto& sub_rule : sub_rules_) {
    body_items.emplace_back();
    body_items.back() << "sub_rule=" << sub_rule->ToDoc();
  }
}

UnionFusionRule::UnionFusionRule(String rule_name, int priority, Array<FusionRule> sub_rules) {
  auto node = runtime::make_object<UnionFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->sub_rules_ = std::move(sub_rules);
  data_ = std::move(node);
}

Array<CandidateKernel> CoalesceFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  Array<CandidateKernel> candidates =
      sub_rule_->AllCandidateKernels(dataflow_graph, spec, name_supply);
  std::vector<CandidateKernel> sorted_candidates(candidates.begin(), candidates.end());
  // TODO(mbs): Quick hack, does not handle overlapping candidates properly.
  // Sort the candidates by their first-inside index.
  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
            [](const CandidateKernel& left, const CandidateKernel& right) {
              return left->sub_graph_->first_inside_index_ < right->sub_graph_->first_inside_index_;
            });
  Array<CandidateKernel> result;
  while (!sorted_candidates.empty()) {
    // Take the first candidate.
    CandidateKernel base = sorted_candidates.front();
    sorted_candidates.erase(sorted_candidates.begin());
    // Try to grow it as much as possible.
    for (auto itr = sorted_candidates.begin(); itr != sorted_candidates.end(); /*no-op*/) {
      CandidateKernel rhs = *itr;
      if (base.MaybeUnionable(rhs)) {
        base = base.DisjointUnion(dataflow_graph, rhs);
        itr = sorted_candidates.erase(itr);
      } else {
        ++itr;
      }
    }
    base = CandidateKernel(NestLabels(rule_name_, base->rule_name_), base->priority_,
                           base->sub_graph_, base->spec_);
    VLOG(1) << "CoalesceFusionRule(" << rule_name_ << ") yields " << base->ToString();
    result.push_back(base);
  }
  return result;
}

void CoalesceFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
}

CoalesceFusionRule::CoalesceFusionRule(String rule_name, int priority, FusionRule sub_rule) {
  auto node = runtime::make_object<CoalesceFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

Array<CandidateKernel> SingletonByKindFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  Array<CandidateKernel> result;
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    auto node = dataflow_graph.index_to_node(index);
    Expr sub_expr = node->ref();
    if (sub_expr->IsInstance<CallNode>()) {
      OpPatternKind kind;
      String label;
      std::tie(kind, label) = SubExprKindAndLabel(sub_expr);
      if (kind <= kOutEWiseFusable) {
        IndexSet inside(dataflow_graph.size(), {index});
        SubGraph sub_graph(dataflow_graph, std::move(inside), kind, std::move(label));
        String rule_name = NestLabels(rule_name_, sub_graph->label_);
        CandidateKernel candidate(std::move(rule_name), priority_, std::move(sub_graph), spec);
        VLOG(1) << "SingletonByKindFusionRule(" << rule_name_ << ") yields "
                << candidate->ToString();
        result.push_back(candidate);
      }
    }
  }
  return result;
}

void SingletonByKindFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
}

SingletonByKindFusionRule::SingletonByKindFusionRule(String rule_name, int priority) {
  auto node = runtime::make_object<SingletonByKindFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  data_ = std::move(node);
}

bool KindSimplePrimRule::Fires(const DataflowGraph& dataflow_graph, const CandidateKernel& upstream,
                               const CandidateKernel& downstream) const {
  return upstream->sub_graph_->kind_ <= upstream_kind_ &&
         downstream->sub_graph_->kind_ <= downstream_kind_;
}

void PrimRuleResults::Add(const CandidateKernel& new_candidate) {
  if (seen.count(new_candidate->sub_graph_)) {
    VLOG(1) << "already seen candidate, ignoring";
  } else {
    seen.emplace(new_candidate->sub_graph_);
    candidates_to_add.emplace_back(new_candidate);
  }
}

void PrimRuleResults::Remove(const CandidateKernel& old_candidate) {
  ICHECK(seen.count(old_candidate->sub_graph_));
  candidates_to_remove.emplace_back(old_candidate);
}

bool PrimRuleResults::PrepareForNextRound() {
  size_t init_size = current_candidates.size();
  for (const auto& candidate_to_remove : candidates_to_remove) {
    current_candidates.erase(
        std::remove(current_candidates.begin(), current_candidates.end(), candidate_to_remove),
        current_candidates.end());
  }
  candidates_to_remove.clear();
  VLOG(1) << "removed " << init_size - current_candidates.size() << " candidates";
  if (candidates_to_add.empty()) {
    // We've reached a fixed point and can stop.
    VLOG(1) << "no new candidates, stopping search";
    return false;
  }
  for (auto& new_candidate : candidates_to_add) {
    current_candidates.push_back(new_candidate);
  }
  VLOG(1) << "added " << candidates_to_add.size() << " candidates";
  candidates_to_add.clear();
  return true;
}

void AllSimplePrimRules::AppendAllResults(const DataflowGraph& dataflow_graph,
                                          PrimRuleResults& results) const {
  for (size_t i = 0; i < results.current_candidates.size(); ++i) {
    CandidateKernel upstream = results.current_candidates[i];
    for (size_t j = 0; j < results.current_candidates.size(); ++j) {
      if (i == j) {
        continue;
      }
      CandidateKernel downstream = results.current_candidates[j];
      if (upstream.MaybeUnionable(downstream)) {
        for (const auto& simple_rule : simple_prim_rules_) {
          if (simple_rule->Fires(dataflow_graph, upstream, downstream)) {
            CandidateKernel new_candidate = upstream.DisjointUnion(dataflow_graph, downstream);
            VLOG(1) << "Fired " << simple_rule->prim_rule_name_ << " on " << upstream->ToString()
                    << " and " << downstream->ToString() << " to yield "
                    << new_candidate->ToString();
            results.Add(new_candidate);
          }
        }
      }
    }
  }
}

void TupleArgPrimRule::AppendAllResults(const DataflowGraph& dataflow_graph,
                                        PrimRuleResults& results) const {
  // The two-step I -> tuple -> I rule.
  for (size_t i = 0; i < results.current_candidates.size(); ++i) {
    CandidateKernel tuple_consumer = results.current_candidates[i];
    if (tuple_consumer->sub_graph_->kind_ > kInjective) {
      continue;
    }
    for (PostDfsIndex input_index : tuple_consumer->sub_graph_->input_) {
      auto node = dataflow_graph.index_to_node(input_index);
      Expr sub_expr = node->ref();
      const auto* tuple_node = sub_expr.as<TupleNode>();
      if (tuple_node == nullptr) {
        continue;
      }
      // The tuple_consumer candidate consumes (at least one) tuple.
      auto tuple_dataflow_node = dataflow_graph.item_to_node(tuple_node);
      // Collect all the possible unions. There may be more than one if different candidates
      // could supply the same tuple field.
      std::vector<std::vector<CandidateKernel>> all_possible_unions;
      all_possible_unions.emplace_back();
      all_possible_unions.back().emplace_back(tuple_consumer);
      // For each tuple field...
      for (auto* tuple_field_dataflow_node : tuple_dataflow_node->inputs_) {
        std::vector<CandidateKernel> to_appends;
        for (size_t j = 0; j < results.current_candidates.size(); ++j) {
          if (i == j) {
            continue;
          }
          CandidateKernel tuple_field_producer = results.current_candidates[j];
          if (tuple_field_producer->sub_graph_->kind_ > kInjective) {
            continue;
          }
          if (!tuple_field_producer->sub_graph_->exit_[tuple_field_dataflow_node->index_]) {
            continue;
          }
          // The tuple_field_producer candidate can provide the tuple field.
          to_appends.emplace_back(tuple_field_producer);
        }
        // If to_appends = [A, B] and we already have possible unions [C, D] and [E, F] then
        // the new possible unions are [C, D, A], [C, D, B], [E, F, A] and [E, F, B].
        if (!to_appends.empty()) {
          std::vector<std::vector<CandidateKernel>> new_all_possible_joins;
          for (const auto& to_append : to_appends) {
            for (const auto& possible_join : all_possible_unions) {
              new_all_possible_joins.emplace_back(possible_join);
              new_all_possible_joins.back().emplace_back(to_append);
            }
          }
          all_possible_unions = std::move(new_all_possible_joins);
        }
      }
      // Actually build the candidates which union according to all_possible_unions.
      for (const auto& possible_union : all_possible_unions) {
        if (possible_union.size() > 1) {
          CandidateKernel new_candidate =
              CandidateKernel::DisjointUnion(dataflow_graph, possible_union);
          std::ostringstream os;
          bool first = true;
          for (const auto& candidate : possible_union) {
            if (first) {
              first = false;
            } else {
              os << ", ";
            }
            os << candidate->ToString();
          }
          VLOG(1) << "Fired tuple rule on {" << os.str() << "} to yield "
                  << new_candidate->ToString();
          results.Add(new_candidate);
        }
      }
    }
  }
}

void ConstantPrimRule::AppendAllResults(const DataflowGraph& dataflow_graph,
                                        PrimRuleResults& results) const {
  for (size_t i = 0; i < results.current_candidates.size(); ++i) {
    CandidateKernel base = results.current_candidates[i];
    IndexSet new_constants(dataflow_graph.size());
    for (PostDfsIndex index : base->sub_graph_->input_) {
      auto node = dataflow_graph.index_to_node(index);
      if (const auto* constant_node = node->ref().as<ConstantNode>()) {
        new_constants.Add(index);
      }
    }
    if (!new_constants.IsZero()) {
      SubGraph sub_graph(dataflow_graph, new_constants, kElemWise, "const");
      CandidateKernel new_const_candidate("", /*priority=*/0, std::move(sub_graph), base->spec_);
      CandidateKernel new_candidate = base.DisjointUnion(dataflow_graph, new_const_candidate);
      VLOG(1) << "Fired const rule on " << new_const_candidate->ToString() << " and "
              << base->ToString() << " to yield " << new_candidate->ToString();
      results.Add(new_candidate);
      results.Remove(base);
    }
  }
}

namespace {
std::vector<std::unique_ptr<PrimRule>> StandardPrimRules() {
  std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules;
  simple_prim_rules.emplace_back(
      std::make_unique<KindSimplePrimRule>("A->B", kOutEWiseFusable, kBroadcast));
  simple_prim_rules.emplace_back(
      std::make_unique<KindSimplePrimRule>("B->R", kBroadcast, kCommReduce));
  simple_prim_rules.emplace_back(
      std::make_unique<KindSimplePrimRule>("I->I", kInjective, kInjective));

  std::vector<std::unique_ptr<PrimRule>> prim_rules;
  prim_rules.emplace_back(std::make_unique<AllSimplePrimRules>(std::move(simple_prim_rules)));
  prim_rules.emplace_back(std::make_unique<TupleArgPrimRule>());
  return prim_rules;
}
}  // namespace

Array<CandidateKernel> CombineByKindFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec, NameSupply* name_supply) const {
  // We'll accumulate all the candidates here, starting with those from the sub-rule.
  // Once a candidate is added to this vector it is immutable.
  Array<CandidateKernel> initial_candidates =
      sub_rule_->AllCandidateKernels(dataflow_graph, spec, name_supply);
  PrimRuleResults rule_results;
  for (const auto& candidate : initial_candidates) {
    rule_results.Add(candidate);
  }

  // TODO(mbs): Hopelessly naive, needs indexing.
  while (rule_results.PrepareForNextRound()) {
    VLOG(1) << "looking for fusion opportunities over " << rule_results.current_candidates.size()
            << " existing candidates";
    for (const auto& prim_rule : prim_rules_) {
      prim_rule->AppendAllResults(dataflow_graph, rule_results);
    }
  }

  Array<CandidateKernel> result;
  for (auto& candidate : rule_results.current_candidates) {
    CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                  candidate->priority_, candidate->sub_graph_, candidate->spec_);
    VLOG(1) << "CombineByKindFusionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.push_back(new_candidate);
  }
  return result;
}

void CombineByKindFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
}

CombineByKindFusionRule::CombineByKindFusionRule(
    String rule_name, int priority, FusionRule sub_rule,
    std::vector<std::unique_ptr<PrimRule>> prim_rules) {
  auto node = runtime::make_object<CombineByKindFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->priority_ = priority;
  node->sub_rule_ = std::move(sub_rule);
  node->prim_rules_ = std::move(prim_rules);
  data_ = std::move(node);
}

Optional<Function> DefaultRewriteSubGraphFunc(const Function& function) { return function; }

FusionSpec::FusionSpec(String spec_name, Target target, FusionRule rule, SubGraphConfig config,
                       TRewriteSubGraphFunc fused_result_func) {
  auto node = runtime::make_object<FusionSpecNode>();
  node->spec_name_ = std::move(spec_name);
  node->target_ = std::move(target);
  node->rule_ = std::move(rule);
  node->fused_result_func_ = std::move(fused_result_func);
  node->config_ = config;
  data_ = std::move(node);
}

Array<CandidateKernel> FusionSpecNode::AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                           NameSupply* name_supply) const {
  Array<CandidateKernel> all_candidates =
      rule_->AllCandidateKernels(dataflow_graph, GetRef<FusionSpec>(this), name_supply);
  Array<CandidateKernel> result;
  for (auto& candidate : all_candidates) {
    if (!candidate->sub_graph_->IsValid(dataflow_graph, config_)) {
      VLOG(1) << "Rejected candidate: not valid for " << candidate->ToString();
      continue;
    }
    ICHECK_EQ(candidate->spec_, GetRef<FusionSpec>(this));
    ICHECK(!candidate->function_.defined());
    Function extracted_function = candidate->sub_graph_->ExtractFunction(dataflow_graph);
    Optional<Function> opt_rewritten_function = fused_result_func_(extracted_function);
    if (!opt_rewritten_function) {
      VLOG(1) << "Rejected candidate: fused_result_func yielded none for " << candidate->ToString();
      continue;
    }
    Function rewritten_function = opt_rewritten_function.value();
    rewritten_function = Downcast<Function>(transform::InferTypeExpr(rewritten_function));
    Map<String, ObjectRef> attrs;
    // The partitioned function must be marked as "Primitive" so we don't attempt to do
    // any further processing within it.
    attrs.Set(attr::kPrimitive, Integer(1));
    Optional<String> opt_compiler = target_->GetAttr("compiler", Optional<String>());
    if (opt_compiler) {
      // Also include the target's "Compiler" attribute on the function so the correct BYOC
      // codegen will take place during lowering.
      attrs.Set(attr::kCompiler, opt_compiler.value());
      // Make sure the kernel has a global name.
      attrs.Set(tvm::attr::kGlobalSymbol,
                String(name_supply->Fresh({(std::string)candidate->rule_name_})));
    }
    rewritten_function = WithAttrs(std::move(rewritten_function), std::move(attrs));
    CandidateKernel new_candidate(NestLabels(spec_name_, candidate->rule_name_),
                                  candidate->priority_, candidate->sub_graph_, candidate->spec_,
                                  std::move(rewritten_function));
    result.push_back(new_candidate);
  }
  return result;
}

std::string FusionSpecNode::ToString() const {
  Doc doc;
  doc << "FusionSpec(" << Doc::NewLine(2);
  std::vector<Doc> body_items;
  body_items.emplace_back();
  body_items.back() << "spec_name=" << Doc::StrLiteral(spec_name_);
  body_items.emplace_back();
  body_items.back() << "target=" << target_->ToDebugString();
  body_items.emplace_back();
  body_items.back() << "rule=" << rule_->ToDoc();
  doc << Doc::Indent(2, Doc::Concat(body_items, Doc::NewLine())) << Doc::NewLine();
  doc << ")";
  return doc.str();
}

/*! \brief Returns fusion spec mimicking TVM FuseOps. */
FusionSpec MakeTVMFusionSpec(Target target) {
  // Build singleton candidates for all calls to ops <= kOutEWiseFusable.
  SingletonByKindFusionRule singleton("", /*priority=*/0);
  // Find fusion groups combining the above according to the TVM fusion rules.
  CombineByKindFusionRule combine("", /*priority=*/0, std::move(singleton), StandardPrimRules());
  SubGraphConfig config;
  config.allow_taps = false;
  config.max_max_depth = 3;
  config.max_outputs = 1;
  return FusionSpec("tvm", std::move(target), std::move(combine), config);
}

/*! \brief Returns fusion rule mimicking one entry in the patterns list passed to the
 * MergeComposite pass. */
FusionRule MakeLabelledDFPatternFusionRule(String rule_name, DFPattern dataflow_pattern,
                                           TPatternPredicate predicate) {
  DFPatternFusionRule pattern_rule("", /*priority=*/0, std::move(dataflow_pattern),
                                   std::move(predicate));
  return CompositeFusionRule(std::move(rule_name), /*priority=*/0, std::move(pattern_rule));
}

/*!
 * \brief Returns fusion spec mimicking AnnotateTarget/MergeCompilerRegions/PartitionGraph
 * for "compiler" attribute of \p target.
 */
FusionSpec MakeOpPredicateBYOCSpec(Target target) {
  Optional<String> opt_compiler = target->GetAttr("compiler", Optional<String>());
  ICHECK(opt_compiler.defined());
  std::string compiler = opt_compiler.value();
  // Build singleton candidates for all calls to supported ops.
  OpPredicateFusionRule singleton("", /*priority=*/0, "target." + compiler);
  // Find fusion groups combining the above according to the TVM fusion rules.
  CombineByKindFusionRule combine("", /*priority=*/0, std::move(singleton), StandardPrimRules());
  SubGraphConfig config;
  config.allow_taps = false;
  config.max_max_depth = 3;
  config.max_outputs = 1;
  return FusionSpec(compiler, std::move(target), std::move(combine), config);
}

/*!
 * \brief Returns fusion spec mimicking
 * MergeComposite/AnnotateTarget/MergeCompilerRegions/PartitionGraph passes for "compiler"
 * attribute of \p target.
 */
FusionSpec MakePatternBYOCSpec(Target target, Array<FusionRule> sub_rules) {
  Optional<String> opt_compiler = target->GetAttr("compiler", Optional<String>());
  ICHECK(opt_compiler.defined());
  std::string compiler = opt_compiler.value();

  // Union all the individual pattern rules.
  UnionFusionRule unioned("", /*priority=*/0, std::move(sub_rules));
  // Find fusion groups combining the above according to the TVM fusion rules.
  CombineByKindFusionRule combine("", /*priority=*/0, std::move(unioned), StandardPrimRules());
  SubGraphConfig config;
  config.allow_taps = false;
  config.max_max_depth = 3;
  config.max_outputs = 1;
  return FusionSpec(compiler, std::move(target), std::move(combine), config);
}

TVM_REGISTER_GLOBAL("relay.collage.make_labelled_dfpattern_fusion_rule")
    .set_body_typed([](String rule_name, DFPattern dataflow_pattern, TPatternPredicate predicate) {
      return MakeLabelledDFPatternFusionRule(std::move(rule_name), std::move(dataflow_pattern),
                                             std::move(predicate));
    });

TVM_REGISTER_GLOBAL("relay.collage.make_tvm_fusion_spec").set_body_typed(MakeTVMFusionSpec);

TVM_REGISTER_GLOBAL("relay.collage.make_op_predicate_byoc_spec")
    .set_body_typed(MakeOpPredicateBYOCSpec);

TVM_REGISTER_GLOBAL("relay.collage.make_pattern_byoc_spec").set_body_typed(MakePatternBYOCSpec);

}  // namespace collage
}  // namespace relay
}  // namespace tvm