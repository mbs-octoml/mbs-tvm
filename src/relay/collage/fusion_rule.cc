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
 * \brief Compositional fusion rules.
 */

#include "./fusion_rule.h"

#include <tvm/relay/transform.h>

#include "./fusion_spec.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

Array<CandidateKernel> FusionRuleNode::AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                           const FusionSpec& spec) const {
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
}

FusionRule::FusionRule(String rule_name) {
  auto node = runtime::make_object<FusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

bool DefaultPatternPredicate(const Expr& matched_sub_expr) { return true; }

Array<CandidateKernel> DFPatternFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
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
    CandidateKernel candidate(std::move(rule_name), std::move(sub_graph), spec);
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

DFPatternFusionRule::DFPatternFusionRule(String rule_name, DFPattern pattern,
                                         TPatternPredicate predicate) {
  auto node = runtime::make_object<DFPatternFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->pattern_ = std::move(pattern);
  node->predicate_ = std::move(predicate);
  data_ = std::move(node);
}

Array<CandidateKernel> OpPredicateFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
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
    CandidateKernel candidate(std::move(rule_name), std::move(sub_graph), spec);
    VLOG(1) << "OpPredicateFusionRule(" << rule_name_ << ") yields " << candidate->ToString();
    result.push_back(candidate);
  }
  return result;
}

/* static */
std::pair<Expr, IndexSet> OpPredicateFusionRuleNode::Extract(const DataflowGraph& dataflow_graph,
                                                             const DataflowGraph::Node* node,
                                                             bool include_const_args) {
  VLOG_CONTEXT << "OpPredicateFusionRule Extract";
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

OpPredicateFusionRule::OpPredicateFusionRule(String rule_name, String attribute) {
  auto node = runtime::make_object<OpPredicateFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->attribute_ = std::move(attribute);
  data_ = std::move(node);
}

Array<CandidateKernel> CompositeFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
  Array<CandidateKernel> candidates = sub_rule_->AllCandidateKernels(dataflow_graph, spec);
  Array<CandidateKernel> result;
  for (const auto& candidate : candidates) {
    CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                  candidate->sub_graph_.WithLabel(dataflow_graph, rule_name_),
                                  candidate->spec_);
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

CompositeFusionRule::CompositeFusionRule(String rule_name, FusionRule sub_rule) {
  auto node = runtime::make_object<CompositeFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

Array<CandidateKernel> UnionFusionRuleNode::AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                                const FusionSpec& spec) const {
  Array<CandidateKernel> result;
  for (const auto& sub_rule : sub_rules_) {
    Array<CandidateKernel> candidates = sub_rule->AllCandidateKernels(dataflow_graph, spec);
    for (const auto& candidate : candidates) {
      CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                    candidate->sub_graph_, candidate->spec_);
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

UnionFusionRule::UnionFusionRule(String rule_name, Array<FusionRule> sub_rules) {
  auto node = runtime::make_object<UnionFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rules_ = std::move(sub_rules);
  data_ = std::move(node);
}

Array<CandidateKernel> MaxCoalesceFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
  Array<CandidateKernel> candidates = sub_rule_->AllCandidateKernels(dataflow_graph, spec);
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
    base = CandidateKernel(NestLabels(rule_name_, base->rule_name_), base->sub_graph_, base->spec_);
    VLOG(1) << "MaxCoalesceFusionRule(" << rule_name_ << ") yields " << base->ToString();
    result.push_back(base);
  }
  return result;
}

void MaxCoalesceFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
}

MaxCoalesceFusionRule::MaxCoalesceFusionRule(String rule_name, FusionRule sub_rule) {
  auto node = runtime::make_object<MaxCoalesceFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

Array<CandidateKernel> OpCallByKindFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
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
        CandidateKernel candidate(std::move(rule_name), std::move(sub_graph), spec);
        VLOG(1) << "OpCallByKindFusionRule(" << rule_name_ << ") yields " << candidate->ToString();
        result.push_back(candidate);
      }
    }
  }
  return result;
}

void OpCallByKindFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
}

OpCallByKindFusionRule::OpCallByKindFusionRule(String rule_name) {
  auto node = runtime::make_object<OpCallByKindFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

void PrimRuleResults::Add(const DataflowGraph& dataflow_graph,
                          const CandidateKernel& new_candidate) {
  if (seen.count(new_candidate->sub_graph_)) {
    VLOG(1) << "already seen candidate, ignoring";
  } else if (!new_candidate->sub_graph_->IsValid(dataflow_graph, *config)) {
    VLOG(1) << "candidate not valid, ignoring";
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
  for (const auto& new_candidate : candidates_to_add) {
    current_candidates.push_back(new_candidate);
  }
  VLOG(1) << "added " << candidates_to_add.size() << " candidates";
  candidates_to_add.clear();
  return true;
}

bool ByKindSimplePrimRule::Fires(const DataflowGraph& dataflow_graph,
                                 const CandidateKernel& upstream,
                                 const CandidateKernel& downstream) const {
  return upstream->sub_graph_->kind_ <= upstream_kind_ &&
         downstream->sub_graph_->kind_ <= downstream_kind_;
}

std::string ByKindSimplePrimRule::ToString() const {
  return KindToString(upstream_kind_) + "->" + KindToString(downstream_kind_);
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
            results.Add(dataflow_graph, new_candidate);
          }
        }
      }
    }
  }
}

std::string AllSimplePrimRules::ToString() const {
  std::ostringstream os;
  os << "AllSimplePrimRules(";
  bool first = true;
  for (const auto& simple : simple_prim_rules_) {
    if (first) {
      first = false;
    } else {
      os << ",";
    }
    os << simple->ToString();
  }
  os << ")";
  return os.str();
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
          results.Add(dataflow_graph, new_candidate);
        }
      }
    }
  }
}

std::string TupleArgPrimRule::ToString() const { return "TupleArgPrimRule()"; }

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
      CandidateKernel new_const_candidate("", std::move(sub_graph), base->spec_);
      CandidateKernel new_candidate = base.DisjointUnion(dataflow_graph, new_const_candidate);
      VLOG(1) << "Fired const rule on " << new_const_candidate->ToString() << " and "
              << base->ToString() << " to yield " << new_candidate->ToString();
      results.Add(dataflow_graph, new_candidate);
      results.Remove(base);
    }
  }
}

std::string ConstantPrimRule::ToString() const { return "ConstantPrimRule()"; }

Array<CandidateKernel> CombineByPrimitivesFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
  // We'll accumulate all the candidates here, starting with those from the sub-rule.
  // Once a candidate is added to this vector it is immutable.
  Array<CandidateKernel> initial_candidates = sub_rule_->AllCandidateKernels(dataflow_graph, spec);
  PrimRuleResults rule_results(&config_);
  for (const auto& candidate : initial_candidates) {
    rule_results.Add(dataflow_graph, candidate);
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
  for (const auto& candidate : rule_results.current_candidates) {
    CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                  candidate->sub_graph_, candidate->spec_);
    VLOG(1) << "CombineByPrimitivesFusionRule(" << rule_name_ << ") yields "
            << new_candidate->ToString();
    result.push_back(new_candidate);
  }
  return result;
}

void CombineByPrimitivesFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
  for (const auto& prim_rule : prim_rules_) {
    body_items.emplace_back();
    body_items.back() << "prim_rule=" << prim_rule->ToString();
  }
  body_items.emplace_back();
  body_items.back() << "config=" << config_.ToString();
}

CombineByPrimitivesFusionRule::CombineByPrimitivesFusionRule(
    String rule_name, FusionRule sub_rule, std::vector<std::unique_ptr<PrimRule>> prim_rules,
    size_t max_max_depth_) {
  auto node = runtime::make_object<CombineByPrimitivesFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  node->prim_rules_ = std::move(prim_rules);
  node->config_.max_max_depth = max_max_depth_;
  // The following two properties are not monotonic. Eg we have intermediate candidates
  // with taps who's disjoint union is tap free.
  node->config_.allow_taps = true;
  node->config_.max_outputs = 0;
  data_ = std::move(node);
}

Array<CandidateKernel> OnlyValidFusionRuleNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph, const FusionSpec& spec) const {
  Array<CandidateKernel> result;
  Array<CandidateKernel> candidates = sub_rule_->AllCandidateKernels(dataflow_graph, spec);
  for (const auto& candidate : candidates) {
    if (!candidate->sub_graph_->IsValid(dataflow_graph, config_)) {
      VLOG(1) << "Ignoring invalid candidate " << candidate->ToString();
    } else {
      CandidateKernel new_candidate(NestLabels(rule_name_, candidate->rule_name_),
                                    candidate->sub_graph_, candidate->spec_);
      VLOG(1) << "OnlyValidFusionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
      result.push_back(new_candidate);
    }
  }
  return result;
}

void OnlyValidFusionRuleNode::AppendBodyItems(std::vector<Doc>& body_items) const {
  FusionRuleNode::AppendBodyItems(body_items);
  body_items.emplace_back();
  body_items.back() << "sub_rule=" << sub_rule_->ToDoc();
  body_items.emplace_back();
  body_items.back() << "config=" << config_.ToString();
}

OnlyValidFusionRule::OnlyValidFusionRule(String rule_name, FusionRule sub_rule,
                                         const SubGraphConfig& config) {
  auto node = runtime::make_object<OnlyValidFusionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  node->config_ = config;
  data_ = std::move(node);
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm