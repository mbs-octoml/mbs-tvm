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
 * \file src/relay/collage/candidate_kernel.cc
 * \brief A potential kernel in the search.
 */

#include "./candidate_kernel.h"

#include <tvm/relay/transform.h>

#include "./fusion_rule.h"
#include "./fusion_spec.h"

namespace tvm {
namespace relay {
namespace collage {

FusionSpec CandidateKernelNode::fusion_spec() const { return Downcast<FusionSpec>(spec_); }

std::string CandidateKernelNode::fusion_spec_name() const {
  return Downcast<FusionSpec>(spec_)->spec_name_;
}

std::string CandidateKernelNode::ToString() const {
  std::ostringstream os;
  os << "{rule_name=" << rule_name_;
  os << ",sub_graph=" << sub_graph_->ToString();
  os << ",spec_name=" << fusion_spec_name();
  if (target_.defined()) {
    os << ",target=" << target_->ToDebugString();
  }
  if (function_.defined()) {
    os << ",function=...";
  }
  if (!cost_.is_unknown()) {
    os << ",cost=" << cost_.ToString();
  }
  os << "}";
  return os.str();
}

Cost CandidateKernelNode::EstimatedCost(const DataflowGraph& dataflow_graph,
                                        CostEstimator* cost_estimator,
                                        NameSupply* name_supply) const {
  ICHECK(target_.defined());
  if (cost_.is_unknown()) {
    VLOG_CONTEXT << "spec " << fusion_spec_name();
    ICHECK(!function_.defined());
    Function extracted_function = sub_graph_->ExtractFunction(dataflow_graph);
    VLOG(1) << "Rewriting function:\n" << PrettyPrint(extracted_function);
    Optional<Function> opt_rewritten_function =
        fusion_spec()->fused_result_func_(extracted_function);
    if (!opt_rewritten_function) {
      cost_ = Cost::Invalid();
      VLOG(1) << "Unable to rewrite function, candidate now " << ToString();
    } else {
      Function rewritten_function = opt_rewritten_function.value();
      rewritten_function = Downcast<Function>(transform::InferTypeExpr(rewritten_function));
      Map<String, ObjectRef> attrs;
      // The extracted function must be marked as "Primitive" so we don't attempt to do
      // any further processing within it.
      attrs.Set(attr::kPrimitive, Integer(1));
      Optional<String> opt_compiler = target_->GetAttr("compiler", Optional<String>());
      if (opt_compiler) {
        // Also include the target's "Compiler" attribute on the function so the correct BYOC
        // codegen will take place during lowering.
        attrs.Set(attr::kCompiler, opt_compiler.value());
        // Make sure the kernel has a unique global name.
        attrs.Set(tvm::attr::kGlobalSymbol, String(name_supply->Fresh({(std::string)rule_name_})));
      }
      function_ = WithAttrs(rewritten_function, std::move(attrs));
      VLOG(1) << "Estimating cost of:\n" << PrettyPrint(function_);
      cost_ = cost_estimator->CachedEstimate(function_, target_);
      VLOG(1) << "Estimated cost, candidate now " << ToString();
    }
  } else {
    VLOG(1) << "Reusing cached cost";
  }
  return cost_;
}

CandidateKernel::CandidateKernel(String rule_name, SubGraph sub_graph,
                                 ObjectRef /* actually FusionSpec */ spec, Target target) {
  auto node = runtime::make_object<CandidateKernelNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_graph_ = std::move(sub_graph);
  node->spec_ = std::move(spec);
  node->target_ = std::move(target);
  // function default to null, set by EstimatedCost
  // cost defaults to unknown, set by EstimatedCost
  data_ = std::move(node);
}

bool CandidateKernel::MaybeUnionable(const CandidateKernel& that) const {
  return get()->sub_graph_.AreDisjoint(that->sub_graph_) &&
         get()->sub_graph_->output_.Intersects(that->sub_graph_->entry_);
}

CandidateKernel CandidateKernel::DisjointUnion(const DataflowGraph& dataflow_graph,
                                               const CandidateKernel& that) const {
  ICHECK_EQ(get()->spec_, that->spec_);
  ICHECK(!get()->function_.defined());
  ICHECK(!that->function_.defined());
  ICHECK(get()->cost_.is_unknown());
  ICHECK(that->cost_.is_unknown());
  return CandidateKernel(get()->rule_name_ + "+" + that->rule_name_,
                         get()->sub_graph_.DisjointUnion(dataflow_graph, that->sub_graph_),
                         get()->spec_);
}

/*static*/
CandidateKernel CandidateKernel::DisjointUnion(const DataflowGraph& dataflow_graph,
                                               std::vector<CandidateKernel> candidates) {
  ICHECK_GT(candidates.size(), 1);
  CandidateKernel result = candidates.front();
  for (size_t i = 1; i < candidates.size(); ++i) {
    result = result.DisjointUnion(dataflow_graph, candidates[i]);
  }
  return result;
}

/*static*/
Expr CandidateKernel::ParallelPartition(std::unique_ptr<DataflowGraph> dataflow_graph, Expr expr,
                                        std::vector<CandidateKernel> candidates) {
  // IMPORTANT:
  //  - We will be iteratively rewriting expr but will not rewrite the dataflow_graph to
  //    match. So we need to keep a handle on expr so that the dataflow_graph does not end up
  //    with dangling pointers (even though technically we'll never dereference them).
  //  - All the sub-graphs will be w.r.t. the dataflow graph for the original expression.
  //    Each time we call SubGraph::Partition on one of those graphs the result expression will
  //    be rewritten from the final output down to the arguments of the newly extracted function.
  //    However the arguments to that extracted function will be shared with the original
  //    expression. Thus it is safe to iteratively partition over all the sub-graphs without
  //    redoing the dataflow_graph and substituting indexes provided we work in reverse dataflow
  //    order.
  // TODO(mbs): ICHECK to confirm? Make sure this is right, and ICHECK
  std::sort(candidates.begin(), candidates.end(),
            [](const CandidateKernel& left, const CandidateKernel& right) {
              return left->sub_graph_->last_inside_index_ > right->sub_graph_->last_inside_index_;
            });
  Expr result = expr;  // copy!
  for (const auto& candidate : candidates) {
    result = candidate->sub_graph_->Partition(*dataflow_graph, result, candidate->function_);
  }
  return result;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm