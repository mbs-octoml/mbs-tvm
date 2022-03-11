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
#include "./candidate_kernel.h"
#include "./sub_graph.h"

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
 *  - \p OpCallByKindFusionRule: Uses the "TOpPattern" attribute provided for every Relay
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
 *  - \p CombineByPrimitivesFusionRule: Given a sub-rule and a list of 'primitive' rules, finds all
 *    possible ways of combining the sub-rule candidates to yield even larger candidates.
 *    Note that the sub-rule's candidates may also be included in the results -- that is every
 *    combination of candidates is considered optional. The 'primitive' rules allow combining by
 *    \p OpPatternKinds, and combining the arguments to tuples which themselves are arguments
 *    to Relay operator calls. This rule is intended to mimic the existing TVM \p FuseOps pass,
 *    though: i) all combinations are found, ii) the starting set of candidates can be provided
 *    by any other rule (ie not just \p OpCallByKindFusionRule), and iii) we rely on \p SubGraph
 *    validity checking to weed out infeasible candidates.
 *  - \p OnlyValidFusionRule: Ignore candidates who's sub-graphs are invalid w.r.t. a given
 *    \p SubGraphConfig. This can be used to limit the maximum candidate depth, the number
 *    of independent outputs, whether intermediate 'taps' are allowed, and so on.
 *
 * Though not yet implemented, we'd like to allow a combinator rule which will union candidate
 * based on their 'anchor' operators. This can be used to implement 'vertical' and 'horizontal'
 * fusion on more primitive candidates. Note that the \p SubGraph machinery supports
 * multiple-input and -output sub-graphs and their validation, so horizontal fusion is easy
 * implement.
 *
 * We also have \p MaxCoalesceFusionRule, which eagerly combines 'touching' candidates (ie
 * candidates where the output of one sub-graph can be directly connected to the input of the other
 * sub-graph) to form the largest possible candidate. The idea is once the search has been completed
 * this rule can be used to collapse adjacent kernels intended for the same target.
 *
 * Here's some typical \p FusionRule combinations for different fusion strategies:
 *  - Classic TVM \p FuseOps
 *    \code
 *         OpCallByKindFusionRule
 *                   |
 *                   v
 *     CombineByPrimitivesFusionRule (with default TVM primitive rules)
 *    \endcode
 *
 *  - Classic operator-based BYOC with \p AnnotateTarget/MergeCompilerRegions/PartitionGraph passes:
 *    \code
 *          OpPredicateFusionRule
 *                    |
 *                    v
 *       CombineByPrimitivesFusionRule (with join anything primitive rule)
 *    \endcode
 *
 *  - Classic pattern-based BYOC with \p MergeComposite/AnnotateTarget/PartitionGraph passes:
 *    \code
 *
 *      DFPatternFusionRule(pattern1) ... DFPatternFusionRule(patternn)
 *                 |                                               |
 *                 v                                               v
 *       CompositeFusionRule(label1)  ...  CompositeFusionRule(labeln)
 *                           \                 /
 *                            v               v
 *                             UnionFusionRule
 *                                    |
 *                                    v
 *                     CombineByPrimitivesFusionRule (with join anything primitive rule)
 *    \endcode
 *
 *  - "Just fuse what I tell you to fuse", using \p DFPatterns to directly select candidates:
 *    \code
 *      DFPatternFusionRule(pattern1)  ...  DFPatternFusionRule(patternn)
 *                           \                   /
 *                            v                 v
 *                               UnionFusionRule
 *    \endcode
 *
 *  - "Consider this library implementation for these sub-expressions", using \p DFPatterns to
 *    pick out which Relay operators are supported (note that TVM lowering does not currently
 *    support this):
 *    \code
 *     OpCallByKindFusionRule    DFPatternFusionRule(pattern1) ... DFPatternFusionRule(patternn)
 *                \                            |                                 |
 *                 \                           v                                 v
 *                  \               CompositeFusionRule(label1)  ...  CompositeFusionRule(labeln)
 *                   \                         |                            /
 *                    v                        v                           v
 *                                       UnionFusionRule
 *                                             |
 *                                             v
 *                               CombineByPrimitivesFusionRule (with default TVM primitive rules)
 *   \endcode
 */
class FusionRuleNode : public Object {
 public:
  /*!
   * \brief A unique (over all rules for the same target) name for the rule.
   */
  String rule_name_;

  void VisitAttrs(AttrVisitor* v) { v->Visit("rule_name", &rule_name_); }

  /*!
   * \brief Returns all the possible candidate kernels for overall expression corresponding to
   * \p dataflow_graph. The candidates will have unknown target, function and cost.
   */
  virtual Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                                     const FusionSpec& spec) const;

  std::string ToString() const;
  Doc ToDoc() const;

 protected:
  virtual void AppendBodyItems(std::vector<Doc>& body_items) const;

 public:
  static constexpr const char* _type_key = "relay.collage.FusionRule";
  static constexpr const uint32_t _type_child_slots = 8;
  TVM_DECLARE_BASE_OBJECT_INFO(FusionRuleNode, Object);
};

class FusionRule : public ObjectRef {
 public:
  FusionRule(String rule_name);

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
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.DFPatternFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(DFPatternFusionRuleNode, FusionRuleNode);
};

class DFPatternFusionRule : public FusionRule {
 public:
  DFPatternFusionRule(String rule_name, DFPattern pattern,
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
                                             const FusionSpec& spec) const override;

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
  OpPredicateFusionRule(String rule_name, String attribute);

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
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.CompositeFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompositeFusionRuleNode, FusionRuleNode);
};

class CompositeFusionRule : public FusionRule {
 public:
  CompositeFusionRule(String rule_name, FusionRule sub_rule);

  TVM_DEFINE_OBJECT_REF_METHODS(CompositeFusionRule, FusionRule, CompositeFusionRuleNode);
};

/*!
 * \brief Fusion rule which simply unions all matches from all sub-fusion rules.
 */
class UnionFusionRuleNode : public FusionRuleNode {
 public:
  Array<FusionRule> sub_rules_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.UnionFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(UnionFusionRuleNode, FusionRuleNode);
};

class UnionFusionRule : public FusionRule {
 public:
  UnionFusionRule(String rule_name, Array<FusionRule> sub_rules);

  TVM_DEFINE_OBJECT_REF_METHODS(UnionFusionRule, FusionRule, UnionFusionRuleNode)
};

/*!
 * \brief Fusion rule which maximally coalesces (using DisjointUnion) candidate kernels from the
 * given sub-fusion rule.
 */
class MaxCoalesceFusionRuleNode : public FusionRuleNode {
 public:
  FusionRule sub_rule_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.MaxCoalesceFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(MaxCoalesceFusionRuleNode, FusionRuleNode);
};

class MaxCoalesceFusionRule : public FusionRule {
 public:
  MaxCoalesceFusionRule(String rule_name, FusionRule sub_rule);

  TVM_DEFINE_OBJECT_REF_METHODS(MaxCoalesceFusionRule, FusionRule, MaxCoalesceFusionRuleNode);
};

/*
 *! \brief Fusion rule which places calls to Relay operators with a "TOpPattern" attribute of
 * \p kOutEWiseFusable or less in their own singleton sub-graph. No other Relay sub-expressions
 * (such as tuples or tuple projection) are selected, and it is up to outer fusion rules to account
 * for them.
 */
class OpCallByKindFusionRuleNode : public FusionRuleNode {
 public:
  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

  static constexpr const char* _type_key = "relay.collage.OpCallByKindFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpCallByKindFusionRuleNode, FusionRuleNode);
};

class OpCallByKindFusionRule : public FusionRule {
 public:
  OpCallByKindFusionRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(OpCallByKindFusionRule, FusionRule, OpCallByKindFusionRuleNode);
};

/*!
 * \brief Holds a vector of current candidates and the additions/removals to apply to them
 * based on the following 'primitive' rules.
 */
struct PrimRuleResults {
  explicit PrimRuleResults(const SubGraphConfig* config) : config(config) {}
  void Add(const DataflowGraph& dataflow_graph, const CandidateKernel& new_candidate);
  void Remove(const CandidateKernel& old_candidate);
  bool PrepareForNextRound();

  const SubGraphConfig* config;
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

  virtual std::string ToString() const = 0;

  std::string prim_rule_name_;
};

/*!
 * \brief A simple primitive rule which fires if the \p upstream and \p downstream candidates have
 * the give \p upstream_kind and \p downstream_kind (or less) respectively.
 */
class ByKindSimplePrimRule : public SimplePrimRule {
 public:
  ByKindSimplePrimRule(std::string sub_rule_name, OpPatternKind upstream_kind,
                       OpPatternKind downstream_kind)
      : SimplePrimRule(std::move(sub_rule_name)),
        upstream_kind_(upstream_kind),
        downstream_kind_(downstream_kind) {}
  ~ByKindSimplePrimRule() override = default;

  bool Fires(const DataflowGraph& dataflow_graph, const CandidateKernel& upstream,
             const CandidateKernel& downstream) const override;
  std::string ToString() const override;


  OpPatternKind upstream_kind_;
  OpPatternKind downstream_kind_;
};

/*!
 * \brief Given the current candidates, apply a rule to add new candidates or remove existing
 * candidates.
 */
class PrimRule {
 public:
  PrimRule() = default;
  virtual ~PrimRule() = default;
  virtual void AppendAllResults(const DataflowGraph& dataflow_graph,
                                PrimRuleResults& results) const = 0;
  virtual std::string ToString() const = 0;
};

/*!
 * \brief Runs one or more simple primitive rules over the current candidates in \p results.
 */
class AllSimplePrimRules : public PrimRule {
 public:
  explicit AllSimplePrimRules(std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules)
      : simple_prim_rules_(std::move(simple_prim_rules)) {}
  ~AllSimplePrimRules() override = default;

  void AppendAllResults(const DataflowGraph& dataflow_graph,
                        PrimRuleResults& results) const override;
  std::string ToString() const override;

  std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules_;
};

/*!
 * \brief Fuses injective sub-groups which appear inside tuples which are themselves inputs to
 * injective sub-groups.
 */
class TupleArgPrimRule : public PrimRule {
 public:
  TupleArgPrimRule() = default;
  ~TupleArgPrimRule() override = default;
  void AppendAllResults(const DataflowGraph& dataflow_graph,
                        PrimRuleResults& results) const override;
  std::string ToString() const override;
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
  std::string ToString() const override;
};

/*!
 * \brief Fusion rule which fuses sub-graphs to exploit optimizations commonly available in backends
 * (including the TVM lowering backend). Those optimization rules are in turn described by one or
 * more \p PrimRules.
 *
 * For TVM these primitive rules are guided by the \p OpPatternKind associated with every sub-graph.
 * That in turn is the maximum of the kind of each expression node in the sub-graph, using the
 * rules:
 *  - Constants are \p kElemwise.
 *  - A call to a Relay operator has the kind of its callee.
 *  - Tuple construction and projection are injective provided all tuple fields are of tensor type.
 *  - All other sub-expressions are opaque.
 *
 * The available \p OpPatternKinds (and our abbreviations for them) are:
 *  - E: kElemWise, eg nn.relu
 *  - B: kBroadcast, eg add
 *  - I: kInjective, eg concatenate
 *  - R: kCommReduce, eg sum
 *  - A: kOutEWiseFusable, eg nn.conv2d (often called 'anchor nodes', hence the A abbreviation)
 *  - O: kOpaque, everything else
 * (The kTuple kind is not used by this machinery.)
 *
 * Kinds are ordered as above from least- to most-constraining w.r.t. possible fusion
 * opportunities. When we write a kind abbreviation below we intend it to mean that kind *or less*.
 * And when when write 'kl -> kr' we mean it to match a sub-expression of kind kr or less who's
 * dataflow inputs are all of kind kl or less.
 *
 * We can then mimic the classic \p FuseOps TVM Pass with the following primitive fusion rules:
 *  - Sub-groups cannot have taps. In the classic \p FuseOps pass taps are avoided by construction
 *    by always considering all node->dominator paths. Here we naively allow taps on all candidates,
 *    but reject them using SubGraph::IsValid with a SubGraphConfig with allow_taps = false.
 *  - Join A -> B
 *  - Join B -> R
 *  - Join I -> I
 *  - Join I -> tuple -> I. That is, if an I sub-graph has a tuple as input, and at least one
 *    tuple field can be provided by an I sub-graph exit, then both the tuple and all such fields
 *    may be joined.
 *
 * Note that \p FuseOps only considers the largest possible sub-graphs. However this fusion rule
 * considers all possibilities so as to 'make room' for other targets supplying other
 * overlapping candidates.
 *
 * Other BYOC toolchains may have different primitive rules, which can be expressed by extending
 * \p PrimRule above.
 */
class CombineByPrimitivesFusionRuleNode : public FusionRuleNode {
 public:
  FusionRule sub_rule_;
  std::vector<std::unique_ptr<PrimRule>> prim_rules_;
  /*! \brief Constraints to apply to all intermediate candidates. */
  SubGraphConfig config_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

 public:
  static constexpr const char* _type_key = "relay.collage.CombineByPrimitivesFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(CombineByPrimitivesFusionRuleNode, FusionRuleNode);
};

class CombineByPrimitivesFusionRule : public FusionRule {
 public:
  CombineByPrimitivesFusionRule(String rule_name, FusionRule sub_rule,
                                std::vector<std::unique_ptr<PrimRule>> prim_rules,
                                size_t max_max_depth_ = 4);

  TVM_DEFINE_OBJECT_REF_METHODS(CombineByPrimitivesFusionRule, FusionRule,
                                CombineByPrimitivesFusionRuleNode);
};

/*!
 * \brief Fusion rules which keeps only candidates from the sub-rule with sub-groups valid
 * w.r.t. the given \p SubGraphConfig.
 */
class OnlyValidFusionRuleNode : public FusionRuleNode {
 public:
  FusionRule sub_rule_;
  SubGraphConfig config_;

  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph,
                                             const FusionSpec& spec) const override;

  void AppendBodyItems(std::vector<Doc>& body_items) const override;

 public:
  static constexpr const char* _type_key = "relay.collage.OnlyValidFusionRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(OnlyValidFusionRuleNode, FusionRuleNode);
};

class OnlyValidFusionRule : public FusionRule {
 public:
  OnlyValidFusionRule(String rule_name, FusionRule sub_rule, const SubGraphConfig& config);

  TVM_DEFINE_OBJECT_REF_METHODS(OnlyValidFusionRule, FusionRule, OnlyValidFusionRuleNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_FUSION_RULE_H_
