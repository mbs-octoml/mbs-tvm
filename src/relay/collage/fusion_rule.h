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
 * \file src/relay/collage/fusion_rule.h
 * \brief Compositional fusion rules.
 */

#ifndef SRC_RELAY_COLLAGE_FUSION_RULE_H_
#define SRC_RELAY_COLLAGE_FUSION_RULE_H_

#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/expr.h>

#include "../../printer/doc.h"
#include "candidate_kernel.h"
#include "name_supply.h"
#include "sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Type of function to check if a matched sub-expression should be accepted by a rule. This
 * can be used to, eg, reject operators of unsupported shape or dtype, or otherwise implement rules
 * which are difficult to express in the dataflow pattern language directly.
 */
using TPatternPredicate = TypedPackedFunc<bool(const Expr& matched_sub_expr)>;

/*!
 * \brief The default pattern predicate. Always returns true.
 */
bool DefaultPatternPredicate(const Expr& matched_sub_expr);

/*! \brief Base class of all fusion rules.
 *
 * A \p FusionRule describes how to find a set of \p CandidateKernels for a \p DataflowGraph.
 * The candidates are allowed to overlap, and ultimately it is the job of the Collage
 * fusion searcher to find a selection of candidates which covers the whole Relay expression
 * without overlap. Fusion rules are paired with their \p Target and other 'top level'
 * configuration in a \p FusionSpec below.
 *
 * We provide a set of 'base' fusion rules which produce candidates from the dataflow graph
 * directly. We also provide a set of 'combinator' rules which can produce new candidates
 * from the results of an arbitrary sub-rule or sub-rules. In this way it is possible to
 * combine the fusion rules to express a wide variety of fusion strategies, akin to the way
 * we can combine TVM passes.
 *
 * There may be many thousands of candidates in flight during the fusion search. We take care
 * to defer rewriting any Relay expressions (eg to extract the fused function, or partition
 * the model) until absolutely necessary.
 *
 * The base rules implemented so far:
 *  - \p DFPatternFusionRule: Given a \p DFPattern and expression predicate, produces a candidate
 *    for every sub-graph matched by the pattern and predicate. Unlike the \p PatternRewriter,
 *    candidates are free to overlap. This is the foundation for pattern-based BYOC integrations,
 *    and can be used to write targeted fusion rules as well as find examples of 'composite'
 *    operators.
 *  - \p OpPredicateFusionRule: Given an attribute name, produces a candidate for every call
 *    to a primitive Relay operator where the operator has predicate bound to that attribute which
 *    returns true given the call sub-expression. Generally this will result in a singleton
 *    sub-graph containing only the call, but it may pull in constant arguments to the call
 *    should they be required. This is the foundation for operator-based BYOC integrations,
 *    though we should consider retiring this mechanism in favor of pattern-based alone.
 *  - \p SingletonByKindFusionRule: Uses the "TOpPattern" attribute provided for every Relay
 *    operator to produce a candidate for every call to a 'fusable Relay operator'. This can
 *    be used as the foundation for generic fusion patterns which work over all Relay operators
 *    with particular properties (elementwise, broadcast, injective, reductive, anchor).
 *
 * The combinator rules implemented so far:
 *  - \p CompositeFusionRule: 'Tags' the candidates matched by an arbitrary sub-rule with the
 *    rule name. Tagged sub-graphs are turned into "Primitive" Function with the "Composite"
 *    attribute bound to the tag. This can be used to indicate Relay operators (or groups of
 *    Relay operators) are to be rewritten to specific target-specific operators. This combinator
 *    wraps the \p DFPatternFusionRules for the pattern-based BYOC integrations. However it could
 *    also be used with the default TVM backend, eg to indicate Relay operators should be
 *    replaced with particular external library implementations.
 *  - \p UnionFusionRule: Simply unions all the candidates from all sub-rules together. This is
 *    how a set of, say, \p DFPatternFusionRules can be combined.
 *  - \p CombineByKindFusionRule: Given a sub-rule and a list of 'primitive' rules, finds all
 *    possible ways of combining the sub-rules candidates to yield even larger candidates.
 *    Note that the sub-rule's candidates are included in the results -- that is every combination
 *    of candidates is considered optional. The 'primitive' rules allow combining by
 *    \p OpPatternKinds, as combining the arguments to tuples which themselves are arguments
 *    to Relay operator calls. This rule is intended to mimic the existing TVM \p FuseOps pass,
 *    though: i) all combinations are found, ii) the starting set of candidates can be provided
 *    by any other rule (ie not just \p SingletonByKindFusionRule), and iii) we rely on \p SubGraph
 *    validity checking to weed out infeasible candidates.
 *
 * Though not yet implemented, we'd like to allow a combinator rule which will union candidate
 * based on their 'anchor' operators. This can be used to implement 'vertical' and 'horizontal'
 * fusion on more primitive candidates. Note that the \p SubGraph machinery supports
 * multiple-input and -output sub-graphs and their validation, so horizontal fusion is easy
 * implement.
 *
 * We also have \p CoalesceFusionRule, which eagerly combines 'touching' candidates (ie candidates
 * where the output of one sub-graph can be directly connected to the input of the other sub-graph)
 * to form the largest possible candidate. The idea is once the search has been completed this
 * rule can be used to collapse adjacent kernels intended for the same target.
 *
 * Here's some typical \p FusionRule combinations for different fusion strategies:
 *  - Classic TVM \p FuseOps
 *    \code
 *       SingletonByKindFusionRule
 *                 |
 *                 v
 *        CombineByKindFusionRule
 *    \endcode
 *
 *  - Classic operator-based BYOC with \p AnnotateTarget/MergeCompilerRegions/PartitionGraph passes:
 *    \code
 *       OpPredicateFusionRule
 *                 |
 *                 v
 *       CombineByKindFusionRule
 *    \endcode
 *
 *  - Classic pattern-based BYOC with \p MergeComposite/AnnotateTarget/PartitionGraph passes:
 *    \code
 *
 *      DFPatternFusionRule(pattern1)          ...     DFPatternFusionRule(patternn)
 *                 |                                               |
 *                 v                                               v
 *       CompositeFusionRule(label1)           ...      CompositeFusionRule(labeln)
 *                     \                                      /
 *                      \                                    /
 *                       v                                  v
 *                                 UnionFusionRule
 *                                       |
 *                                       v
 *                             CombineByKindFusionRule
 *    \endcode
 *
 *  - "Just fuse what I tell you to fuse", using \p DFPatterns to directly select candidates:
 *    \code
 *      DFPatternFusionRule(pattern1)          ...     DFPatternFusionRule(patternn)
 *                 |                                               |
 *                 v                                               v
 *                                 UnionFusionRule
 *    \endcode
 *
 *  - "Consider this library implementation for these sub-expressions", using \p DFPatterns to
 *    pick out which Relay operators are supported (note that TVM lowering does not currently
 *    support this):
 *    \code
 *     SingletonByKindFusionRule   DFPatternFusionRule(pattern1) ... DFPatternFusionRule(patternn)
 *                \                            |                                 |
 *                 \                           v                                 v
 *                  \               CompositeFusionRule(label1)  ...  CompositeFusionRule(labeln)
 *                   \                         |                                 |
 *                    v                        v                                 v
 *                                       UnionFusionRule
 *                                             |
 *                                             v
 *                                   CombineByKindFusionRule
 *   \endcode
 */
class FusionRuleNode : public Object {
 public:
  /*!
   * \brief A unique (over all rules for the same target) name for the rule.
   */
  String rule_name_;

  /*!
   * \brief Global priority of this rule w.r.t. all others, including for other targets.
   * Ties between kernels with indistinguishable costs are broken by priority, where higher
   * wins.
   */
  int priority_ = 0;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("rule_name", &rule_name_);
    v->Visit("priority", &priority_);
  }

  /*!
   * \brief Returns all the possible candidate kernels for overall expression corresponding to
   * \p dataflow_graph for target implied by \p spec. The candidates will have unknown cost
   * until it is explicitly estimated.
   */
  virtual Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                     const FusionSpec& spec,
                                                     NameSupply* name_supply) const;

  std::string ToString() const;
  Doc ToDoc() const;

 protected:
  virtual void AppendBodyItems(std::vector<Doc>& body_items) const;

 public:
  static constexpr const char* _type_key = "relay.collage.FusionRule";
  static constexpr const uint32_t _type_child_slots = 7;
  TVM_DECLARE_BASE_OBJECT_INFO(FusionRuleNode, Object);
};

class FusionRule : public ObjectRef {
 public:
  FusionRule(String rule_name, int priority);

  TVM_DEFINE_OBJECT_REF_METHODS(FusionRule, ObjectRef, FusionRuleNode);
};

/*!
 * \brief Fusion rule which fires on all sub-expressions matching a dataflow-pattern and pattern
 * predicate.
 */
class DFPatternFusionRuleNode : public FusionRuleNode {
 public:
  /*!
   * \brief Relay pattern.
   */
  DFPattern pattern_;

  /*!
   * \brief Predicate on matched sub-expression to decide if fusion rule should fire.
   */
  TPatternPredicate predicate_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.DFPatternFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(DFPatternFusionRuleNode, FusionRuleNode);
};

class DFPatternFusionRule : public FusionRule {
 public:
  DFPatternFusionRule(String rule_name, int priority, DFPattern pattern,
                      TPatternPredicate predicate = DefaultPatternPredicate);

  TVM_DEFINE_OBJECT_REF_METHODS(DFPatternFusionRule, FusionRule, DFPatternFusionRuleNode);
};

/*!
 * \brief Fusion rule which fires if a predicate bound to the operator with the given attribute name
 * returns true.
 */
class OpPredicateFusionRuleNode : public FusionRuleNode {
 public:
  /*! Name of operator attribute. If bound it should be to a function of type TPatternPredicate
   * (which is equivalent to the \p FTVMAnnotateTarget function type). */
  String attribute_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

 private:
  /*!
   * \brief Returns the sub-expression rooted at \p node within the overall expression as a
   * stand-alone sub-expression, with free variables representing arbitrary sub-sub-expressions.
   * Also returns the index set representing that sub-expression.
   *
   * If \p include_consts is true include any immediate constant arguments. Otherwise only the
   * immediate sub-expression (ie a Call) is included.
   *
   * This is used to ensure a candidate kernel includes not only a call to a supported operator
   * but also any arguments to that call necessary for the predicate to pass.
   */
  static std::pair<Expr, IndexSet> Extract(const DataflowGraph& dataflow_graph,
                                           const DataflowGraph::Node* node,
                                           bool include_const_args);

 public:
  static constexpr const char* _type_key = "relay.collage.OpPredicateFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpPredicateFusionRuleNode, FusionRuleNode);
};

class OpPredicateFusionRule : public FusionRule {
 public:
  OpPredicateFusionRule(String rule_name, int priority, String attribute);

  TVM_DEFINE_OBJECT_REF_METHODS(OpPredicateFusionRule, FusionRule, OpPredicateFusionRuleNode);
};

/*!
 * \brief Fusion rule which encapsulates candidates matched by a sub-fusion rule within a
 * "Primitive" Relay function tagged with the candidate's rule name as the "Candidate" attributes.
 * This is the standard way by which operators or operator groups are tagged as being supported for
 * a particular BYOC toolchain. The toolchain's code generator relies on the "Composite" attribute
 * to guide which backend operator to invoke.
 */
class CompositeFusionRuleNode : public FusionRuleNode {
 public:
  /*! \brief The sub-fusion rule. */
  FusionRule sub_rule_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.CompositeFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompositeFusionRuleNode, FusionRuleNode);
};

class CompositeFusionRule : public FusionRule {
 public:
  CompositeFusionRule(String rule_name, int priority, FusionRule sub_rule);

  TVM_DEFINE_OBJECT_REF_METHODS(CompositeFusionRule, FusionRule, CompositeFusionRuleNode);
};

/*!
 * \brief Fusion rule which simply unions all matches from all sub-fusion rules.
 */
class UnionFusionRuleNode : public FusionRuleNode {
 public:
  Array<FusionRule> sub_rules_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.UnionFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnionFusionRuleNode, FusionRuleNode);
};

class UnionFusionRule : public FusionRule {
 public:
  UnionFusionRule(String rule_name, int priority, Array<FusionRule> sub_rules);

  TVM_DEFINE_OBJECT_REF_METHODS(UnionFusionRule, FusionRule, UnionFusionRuleNode)
};

/*!
 * \brief Fusion rule which coalesces (using DisjointUnion) candidate kernels from the given
 * sub-fusion rule.
 */
class CoalesceFusionRuleNode : public FusionRuleNode {
 public:
  FusionRule sub_rule_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.MaximalFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(CoalesceFusionRuleNode, FusionRuleNode);
};

class CoalesceFusionRule : public FusionRule {
 public:
  CoalesceFusionRule(String rule_name, int priority, FusionRule sub_rule);

  TVM_DEFINE_OBJECT_REF_METHODS(CoalesceFusionRule, FusionRule, CoalesceFusionRuleNode);
};

/*
 *! \brief Fusion rule which places calls to Relay operators with a "TOpPattern" attribute of
 * \p kOutEWiseFusable or less in their own singleton sub-graph. No other Relay sub-expressions
 * (such as tuples or tuple projection) are selected, and it is up to outer fusion rules to account
 * for them.
 */
class SingletonByKindFusionRuleNode : public FusionRuleNode {
 public:
  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.SingletonByKindFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(SingletonByKindFusionRuleNode, FusionRuleNode);
};

class SingletonByKindFusionRule : public FusionRule {
 public:
  SingletonByKindFusionRule(String rule_name, int priority);

  TVM_DEFINE_OBJECT_REF_METHODS(SingletonByKindFusionRule, FusionRule,
                                SingletonByKindFusionRuleNode);
};

/*!
 * \brief Holds a vector of current candidates and the additions/removals to apply to them
 * based on the following 'simple' rules.
 */
class PrimRuleResults {
 public:
  PrimRuleResults() = default;
  void Add(const CandidateKernel& new_candidate);
  void Remove(const CandidateKernel& old_candidate);
  bool PrepareForNextRound();

  std::vector<CandidateKernel> current_candidates;
  std::vector<CandidateKernel> candidates_to_add;
  std::vector<CandidateKernel> candidates_to_remove;
  std::unordered_set<SubGraph, SubGraphHash, SubGraphEqual> seen;
};

/*!
 * \brief Given \p upstream and \p downstream candidates which touch, returns true if the
 * an additional union candidate should be included.
 */
class SimplePrimRule {
 public:
  explicit SimplePrimRule(std::string prim_rule_name)
      : prim_rule_name_(std::move(prim_rule_name)) {}
  virtual ~SimplePrimRule() = default;

  virtual bool Fires(const DataflowGraph& dataflow_graph, const CandidateKernel& upstream,
                     const CandidateKernel& downstream) const = 0;
  std::string prim_rule_name_;
};

/*!
 * \brief A simple rule which fires if the \p upstream and \p downstream candidates have
 * the give \p upstream_kind and \p downstream_kind (or less) respectively.
 */
class KindSimplePrimRule : public SimplePrimRule {
 public:
  KindSimplePrimRule(std::string sub_rule_name, OpPatternKind upstream_kind,
                     OpPatternKind downstream_kind)
      : SimplePrimRule(std::move(sub_rule_name)),
        upstream_kind_(upstream_kind),
        downstream_kind_(downstream_kind) {}
  ~KindSimplePrimRule() override = default;

  bool Fires(const DataflowGraph& dataflow_graph, const CandidateKernel& upstream,
             const CandidateKernel& downstream) const override;

  OpPatternKind upstream_kind_;
  OpPatternKind downstream_kind_;
};

/*!
 * \brief A primitive rule to apply over the current set of candidates in \p results.
 */
class PrimRule {
 public:
  PrimRule() = default;
  virtual ~PrimRule() = default;
  virtual void AppendAllResults(const DataflowGraph& dataflow_graph,
                                PrimRuleResults& results) const = 0;
};

/*!
 * \brief Runs one or more simple rules over the current candidates in \p results.
 */
class AllSimplePrimRules : public PrimRule {
 public:
  explicit AllSimplePrimRules(std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules)
      : simple_prim_rules_(std::move(simple_prim_rules)) {}
  ~AllSimplePrimRules() override = default;

  void AppendAllResults(const DataflowGraph& dataflow_graph,
                        PrimRuleResults& results) const override;
  std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules_;
};

/*!
 * \brief Fuses injective sub-groups appear inside tuples which are inputs to injective sub-groups.
 */
class TupleArgPrimRule : public PrimRule {
 public:
  TupleArgPrimRule() = default;
  ~TupleArgPrimRule() override = default;
  void AppendAllResults(const DataflowGraph& dataflow_graph,
                        PrimRuleResults& results) const override;
};

/*!
 * \brief Moves constants into argument positions. Note that scalars are always inlined, so
 * this rule only moves tensor constants.
 */
class ConstantPrimRule : public PrimRule {
 public:
  ConstantPrimRule() = default;
  ~ConstantPrimRule() override = default;
  void AppendAllResults(const DataflowGraph& dataflow_graph,
                        PrimRuleResults& results) const override;
};

/*!
 * \brief Fusion rule which fuses sub-graphs to exploit optimizations available in the TVM
 * lowering pipeline. This is guided by the \p OpPatternKind associated with every sub-graph.
 * That in turn is the maximum kind for each node in the sub-graph, using the rules:
 *  - Constants are \p kElemwise.
 *  - A call to a Relay operator has the kind of its callee.
 *  - Tuple construction and projection are injective provided all tuple fields are of tensor type.
 *  - All other sub-expressions are opaque.
 *
 * We generally mimic the classic FuseOps TVM Pass, but emit all possibilities instead of just
 * the unique largest possible sub-graph.
 *
 * The available \p OpPatternKinds and our abbreviations are:
 *  - E: kElemWise, eg relu
 *  - B: kBroadcast, eg add
 *  - I: kInjective, eg concatenate
 *  - R: kCommReduce, eg sum
 *  - A: kOutEWiseFusable, eg conv2d (often called 'anchor nodes', hence the A abbreviation)
 *  - T: kTuple, ignored here (but used in FuseOps)
 *  - O: kOpaque, everything else
 *
 * Kinds are ordered as above from least- to most-constraining w.r.t. possible fusion
 * opportunities. When we write a kind abbreviation below we intend it to mean that kind *or less*.
 * And when when write 'kl -> kr' we mean it to match a sub-expression of kind kr or less who's
 * dataflow inputs are all of kind kl or less.
 *
 * The fusion rules are then:
 *  - Sub-groups cannot have taps. (The classic FuseOps pass works in terms of node->dominator
 *    paths to avoid taps in any candidates. We prefer instead to just build all the candidates
 *    and reject those with taps at the end.)
 *  - Join A -> B
 *  - Join B -> R
 *  - Join I -> I
 *  - Join I -> tuple -> I. That is, if an I sub-graph has a tuple as input, and at least one
 *    tuple field can be provided by an I sub-graph exit, then both the tuple and all such fields
 *    may be joined.
 *
 * For a flow such as A->B->R the classic FuseOps considers only (A->B)->R, however the above
 * rules also allow A->(B->R).
 *
 * (Note that we don't need to distinguish elemwise and broadcast kinds).
 */
class CombineByKindFusionRuleNode : public FusionRuleNode {
 public:
  FusionRule sub_rule_;
  std::vector<std::unique_ptr<PrimRule>> prim_rules_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec,
                                             NameSupply* name_supply) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

 public:
  static constexpr const char* _type_key = "relay.collage.CombineByKindFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(CombineByKindFusionRuleNode, FusionRuleNode);
};

class CombineByKindFusionRule : public FusionRule {
 public:
  CombineByKindFusionRule(String rule_name, int priority, FusionRule sub_rule,
                          std::vector<std::unique_ptr<PrimRule>> prim_rules);

  TVM_DEFINE_OBJECT_REF_METHODS(CombineByKindFusionRule, FusionRule, CombineByKindFusionRuleNode);
};

/*!
 * \brief Type of functions for checking and optionally rewriting candidate fused functions
 * before they proceed to lowering and codegen. The argument is the function extracted from
 * the overall expression to represent the kernel. The result is one of:
 *  - nullptr if the candidate should be rejected, eg because it has an unsupported shape.
 *    It's often easier to reject candidates when looking at the overall sub-graph rather
 *    than trying to make the selecting patterns exact.
 *  - The argument, indicating the candidate should proceed as is.
 *  - A new function, indicating we wish to rewrite the sub-graph before proceeding. This can be
 *    used to, eg, rewrite to a target-specific operator so as to simplify downstream processing.
 */
using TRewriteSubGraphFunc = TypedPackedFunc<Optional<Function>(const Function& function)>;

/*!
 * \brief The default rewriting function. Simply returns its argument.
 */
Optional<Function> DefaultRewriteSubGraphFunc(const Function& function);

/*!
 * \brief Pairs a \p FusionRule with the \p Target it is intended for. We also allow the
 * each candidate kernel function to be rewritten before the candidate is used for estimating
 * kernel latency or included in the final 'parititoned' Relay expression.
 */
class FusionSpecNode : public Object {
 public:
  /*!
   * \brief Specification name to distinguish this spec from all others. Typcially
   * the BYOC compiler name or "tvm".
   */
  String spec_name_;
  /*!
   * \brief The target for all candidate kernels.
   */
  Target target_;

  /*!
   * \brief The pattern rule to use to gather candidate kernels.
   */
  FusionRule rule_;

  /*!
   * \brief A function for processing the candidate kernel functions before proceeding.
   */
  TRewriteSubGraphFunc fused_result_func_ = DefaultRewriteSubGraphFunc;

  /*!
   * \brief Configuration for checking which sub-graphs are considered valid.
   */
  SubGraphConfig config_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             NameSupply* name_supply) const;

  std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.FusionSpec";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionSpecNode, Object);
};

class FusionSpec : public ObjectRef {
 public:
  FusionSpec(String spec_name, Target target, FusionRule rule, SubGraphConfig config,
             TRewriteSubGraphFunc fused_result_func = DefaultRewriteSubGraphFunc);

  TVM_DEFINE_OBJECT_REF_METHODS(FusionSpec, ObjectRef, FusionSpecNode);
};

FusionRule MakeDFPatternFusionRule(String rule_name, DFPattern dataflow_pattern,
                                   runtime::PackedFunc predicate, Target target);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_FUSION_RULE_H_
