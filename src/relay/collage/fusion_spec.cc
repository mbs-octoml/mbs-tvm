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
 * \file src/relay/collage/fusion_spec.cc
 * \brief Combine a \p FusionRule with one or more \p Targets.
 */

#include "./fusion_spec.h"

#include <tvm/relay/transform.h>

#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

Optional<Function> DefaultRewriteSubGraphFunc(const Function& function) { return function; }

FusionSpec::FusionSpec(String spec_name, Array<Target> targets, FusionRule rule,
                       TRewriteSubGraphFunc fused_result_func) {
  auto node = runtime::make_object<FusionSpecNode>();
  node->spec_name_ = std::move(spec_name);
  node->targets_ = std::move(targets);
  node->rule_ = std::move(rule);
  node->fused_result_func_ = std::move(fused_result_func);
  data_ = std::move(node);
}

Array<CandidateKernel> FusionSpecNode::AllCandidateKernels(
    const DataflowGraph& dataflow_graph) const {
  // Gather all the candidates. They'll have no target, function or cost at this stage.
  Array<CandidateKernel> all_candidates =
      rule_->AllCandidateKernels(dataflow_graph, GetRef<FusionSpec>(this));
  Array<CandidateKernel> result;
  for (auto& candidate : all_candidates) {
    ICHECK_EQ(candidate->spec_, GetRef<FusionSpec>(this));
    ICHECK(!candidate->target_.defined());
    ICHECK(!candidate->function_.defined());
    ICHECK(candidate->cost_.is_unknown());
    // Emit a copy of the candidate for each possible target.
    for (const auto& target : targets_) {
      CandidateKernel new_candidate(NestLabels(spec_name_, candidate->rule_name_),
                                    candidate->sub_graph_, candidate->spec_, target);
      result.push_back(new_candidate);
    }
  }
  return result;
}

std::string FusionSpecNode::ToString() const {
  Doc doc;
  doc << "FusionSpec(" << Doc::NewLine(2);
  std::vector<Doc> body_items;
  body_items.emplace_back();
  body_items.back() << "spec_name=" << Doc::StrLiteral(spec_name_);
  for (const auto& target : targets_) {
    body_items.emplace_back();
    body_items.back() << "target=" << target->ToDebugString();
  }
  body_items.emplace_back();
  body_items.back() << "rule=" << rule_->ToDoc();
  doc << Doc::Indent(2, Doc::Concat(body_items, Doc::NewLine())) << Doc::NewLine();
  doc << ")";
  return doc.str();
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
