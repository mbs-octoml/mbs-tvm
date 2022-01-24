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

#include "./fusion_rule.h"

namespace tvm {
namespace relay {
namespace collage {

FusionSpec CandidateKernelNode::fusion_spec() const { return Downcast<FusionSpec>(spec_); }

std::string CandidateKernelNode::fusion_spec_name() const {
  return Downcast<FusionSpec>(spec_)->spec_name_;
}

Target CandidateKernelNode::target() const { return Downcast<FusionSpec>(spec_)->target_; }

std::string CandidateKernelNode::ToString() const {
  std::ostringstream os;
  os << "{rule_name=" << rule_name_;
  os << ",sub_graph=" << sub_graph_->ToString();
  if (spec_.defined()) {
    os << ",spec_name=" << fusion_spec_name();
  }
  if (!cost_.is_unknown()) {
    os << ",cost=" << cost_.ToString();
  }
  os << "}";
  return os.str();
}

Cost CandidateKernelNode::EstimatedCost(CostEstimator* cost_estimator) const {
  if (cost_.is_unknown()) {
    VLOG(1) << "Estimating cost for spec " << fusion_spec_name() << " of:\n"
            << PrettyPrint(function_);
    {
      VLOG_CONTEXT << "spec " << fusion_spec_name();
      cost_ = cost_estimator->CachedEstimate(function_, target());
    }
    VLOG(1) << "Estimated cost is " << cost_.ToString();
  } else {
    VLOG(1) << "Reusing cached cost " << cost_.ToString();
  }
  return cost_;
}

Expr CandidateKernelNode::Partition(const DataflowGraph& dataflow_graph, const Expr& expr) const {
  return sub_graph_->Partition(dataflow_graph, expr, function_);
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
  return CandidateKernel(
      get()->rule_name_ + "+" + that->rule_name_, std::max(get()->priority_, that->priority_),
      get()->sub_graph_.DisjointUnion(dataflow_graph, that->sub_graph_), get()->spec_);
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

}  // namespace collage
}  // namespace relay
}  // namespace tvm