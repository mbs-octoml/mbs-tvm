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
 * \file src/relay/collage/gather_fusion_specs.cc
 * \brief Gather the relevant \p FusionSpecs from the available \p Targets given by a
 * \p CompilationConfig.
 */

#include "./gather_fusion_specs.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {
/*! \brief Returns the primitive rules mimic's \p FuseOps. */
std::vector<std::unique_ptr<PrimRule>> TVMPrimRules() {
  std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules;
  simple_prim_rules.emplace_back(
      std::make_unique<ByKindSimplePrimRule>("A->B", kOutEWiseFusable, kBroadcast));
  simple_prim_rules.emplace_back(
      std::make_unique<ByKindSimplePrimRule>("B->R", kBroadcast, kCommReduce));
  simple_prim_rules.emplace_back(
      std::make_unique<ByKindSimplePrimRule>("I->I", kInjective, kInjective));

  std::vector<std::unique_ptr<PrimRule>> prim_rules;
  prim_rules.emplace_back(std::make_unique<AllSimplePrimRules>(std::move(simple_prim_rules)));
  prim_rules.emplace_back(std::make_unique<TupleArgPrimRule>());
  return prim_rules;
}

/*! \brief Returns the primitive rules which allow for any touching candidates to be fused. */
std::vector<std::unique_ptr<PrimRule>> DefaultBYOCPrimRules() {
  std::vector<std::unique_ptr<SimplePrimRule>> simple_prim_rules;
  simple_prim_rules.emplace_back(
      std::make_unique<ByKindSimplePrimRule>("A->A", kOutEWiseFusable, kOutEWiseFusable));
  std::vector<std::unique_ptr<PrimRule>> prim_rules;
  prim_rules.emplace_back(std::make_unique<AllSimplePrimRules>(std::move(simple_prim_rules)));
  return prim_rules;
}

/*! \brief Returns fusion rule mimicking TVM FuseOps. */
FusionRule MakeTVMFusionRule() {
  // Build singleton candidates for all calls to ops <= kOutEWiseFusable.
  OpCallByKindFusionRule op_call_by_kind("");
  // Find fusion groups combining the above according to the TVM fusion rules.
  CombineByPrimitivesFusionRule combine("", std::move(op_call_by_kind), TVMPrimRules());
  SubGraphConfig sub_graph_config;
  sub_graph_config.allow_taps = false;
  sub_graph_config.max_max_depth = 3;
  sub_graph_config.max_outputs = 1;
  return OnlyValidFusionRule("", std::move(combine), sub_graph_config);
}

/*! \brief Returns fusion rule mimicking one entry in the patterns list passed to the
 * MergeComposite pass. */
FusionRule MakeLabelledDFPatternFusionRule(String rule_name, DFPattern dataflow_pattern,
                                           TPatternPredicate predicate) {
  DFPatternFusionRule pattern_rule("", std::move(dataflow_pattern), std::move(predicate));
  return CompositeFusionRule(std::move(rule_name), std::move(pattern_rule));
}

/*!
 * \brief Returns fusion rule mimicking AnnotateTarget/MergeCompilerRegions/PartitionGraph
 * for "compiler" attribute of \p target.
 */
FusionRule MakeOpPredicateBYOCFusionRule(const String& compiler) {
  // Build singleton candidates for all calls to supported ops.
  OpPredicateFusionRule singleton("", "target." + compiler);
  // Find fusion groups combining any touching non-opaque candidates
  CombineByPrimitivesFusionRule combine("", std::move(singleton), DefaultBYOCPrimRules());
  SubGraphConfig sub_graph_config;
  sub_graph_config.allow_taps = false;
  sub_graph_config.max_max_depth = 3;
  sub_graph_config.max_outputs = 1;
  return OnlyValidFusionRule("", std::move(combine), sub_graph_config);
}

/*!
 * \brief Returns fusion rule mimicking
 * MergeComposite/AnnotateTarget/MergeCompilerRegions/PartitionGraph passes for "compiler"
 * attribute of \p target.
 */
FusionRule MakePatternBYOCFusionRule(const String& compiler, Array<FusionRule> sub_rules) {
  // Union all the individual pattern rules.
  UnionFusionRule unioned("", std::move(sub_rules));
  // Find fusion groups combining any touching non-opaque candidates.
  CombineByPrimitivesFusionRule combine("", std::move(unioned), DefaultBYOCPrimRules());
  SubGraphConfig sub_graph_config;
  sub_graph_config.allow_taps = false;
  sub_graph_config.max_max_depth = 3;
  sub_graph_config.max_outputs = 1;
  return OnlyValidFusionRule("", std::move(combine), sub_graph_config);
}

TVM_REGISTER_GLOBAL("relay.collage.make_labelled_dfpattern_fusion_rule")
    .set_body_typed([](String rule_name, DFPattern dataflow_pattern, TPatternPredicate predicate) {
      return MakeLabelledDFPatternFusionRule(std::move(rule_name), std::move(dataflow_pattern),
                                             std::move(predicate));
    });

TVM_REGISTER_GLOBAL("relay.collage.make_op_predicate_byoc_fusion_rule")
    .set_body_typed(MakeOpPredicateBYOCFusionRule);

TVM_REGISTER_GLOBAL("relay.collage.make_pattern_byoc_fusion_rule")
    .set_body_typed(MakePatternBYOCFusionRule);

}  // namespace

Array<FusionSpec> GatherFusionSpecs(const CompilationConfig& config) {
  // First collect the fusion rules by 'toolchain' (ie BYOC compiler name or the native "tvm").
  // We'll assume but not verify rules derived from targets are uniquely determined by the
  // toolchain name.
  std::unordered_map<std::string, FusionRule> toolchain_to_rule;
  std::unordered_map<std::string, Array<Target>> toolchain_to_targets;
  auto make_rule = [&toolchain_to_rule, &toolchain_to_targets](const Target& target) {
    Optional<String> opt_compiler = target->GetAttr("compiler", Optional<String>());
    std::string toolchain = opt_compiler.defined() ? opt_compiler.value() : "tvm";
    auto itr = toolchain_to_rule.find(toolchain);
    if (itr != toolchain_to_rule.end()) {
      return itr->second;
    }
    FusionRule rule;
    Optional<FusionRule> opt_rule = target->GetAttr("fusion_rule", Optional<FusionRule>());
    if (opt_rule) {
      rule = opt_rule.value();
      VLOG(1) << "Target " << target->ToDebugString() << " has toolchain " << toolchain
              << " and explicit 'fusion_rule' attribute:\n"
              << rule->ToString();
    } else if (opt_compiler.defined()) {
      // Transition to the Python side so we can get access to the BYOC pattern registry.
      // That will bounce right back into the above construction helpers.
      static const runtime::PackedFunc* make_byoc_fusion_rule =
          runtime::Registry::Get("tvm.relay.collage.make_byoc_fusion_rule");
      ICHECK(make_byoc_fusion_rule);
      rule = (*make_byoc_fusion_rule)(opt_compiler.value());
      VLOG(1) << "Target " << target->ToDebugString() << " is for BYOC toolchain " << toolchain
              << " and has default fusion rule:\n"
              << rule->ToString();
    } else {
      rule = MakeTVMFusionRule();
      VLOG(1) << "Target " << target->ToDebugString()
              << " is for native TVM toolchain and has default fusion rule:\n"
              << rule->ToString();
    }
    toolchain_to_rule.emplace(toolchain, rule);
    toolchain_to_targets[toolchain].push_back(target);
    return rule;
  };

  // Gather all the fusion rules we'll need.
  for (const auto& target : config->primitive_targets) {
    FusionRule rule = make_rule(target);
  }
  // Now group targets with their fusion rules.
  Array<FusionSpec> result;
  for (const auto& kv : toolchain_to_rule) {
    result.push_back(FusionSpec(kv.first, toolchain_to_targets[kv.first], kv.second));
  }
  return result;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm