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

#ifndef SRC_RELAY_COLLAGE_CANDIDATE_KERNEL_H_
#define SRC_RELAY_COLLAGE_CANDIDATE_KERNEL_H_

#include <tvm/runtime/container/string.h>

#include "cost.h"
#include "cost_estimator.h"
#include "sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

class FusionSpec;

/*!
 * \brief A candidate kernel w.r.t. the body of a global function in an \p IRModule.
 */
class CandidateKernelNode : public Object {
 public:
  CandidateKernelNode() = default;

  /*! \brief Name of the fusion rule(s) which produced this candidate. */
  String rule_name_;

  /*!
   * \brief Priority of the fusion rule(s) which produced this candidate.
   */
  int priority_ = 0;

  /*! \brief The sub-graph of the overall expression matched by the fusion rule. */
  SubGraph sub_graph_;

  /*!
   * \brief The fusion specification which produced this candidate.
   */
  ObjectRef /* actually FusionSpec */ spec_;

  /*!
   * \brief The Relay "Primitive" function which represents the above sub-graph,
   * including any additional attributes needed to guide lowering and codegen.
   *
   * Initially null, filled in by the FusionSpec processing.
   */
  Function function_;

  /*!
   * \brief The cost of the kernel.
   *
   * Initially Cost::Unknown, filled in by EstimateCost.
   */
  mutable Cost cost_ = Cost::Unknown();

  /*!
   * \brief Returns the fusion specification which produced this candidate.
   */
  FusionSpec fusion_spec() const;

  /*!
   * \brief Returns the name of the fusion specification which produced this candidate.
   */
  std::string fusion_spec_name() const;

  /*!
   * \brief Returns the target for compiling this candidate.
   */
  Target target() const;

  /*!
   * \brief Return the estimated cost of the candidate kernel, using \p cost_estimator if
   * the cost is not already known.
   */
  Cost EstimatedCost(CostEstimator* cost_estimator) const;

  /*!
   * \brief Returns \p expr rewritten to partition the candidate kernel's sub-graph into an
   * inline "Primitive" function with the appropriate additional annotations.
   */
  Expr Partition(const DataflowGraph& dataflow_graph, const Expr& expr) const;

  std::string ToString() const;
};

class CandidateKernel : public ObjectRef {
 public:
  CandidateKernel(String rule_name, int priority, SubGraph sub_graph,
                  ObjectRef /* actually FusionSpec */ spec, Function function = {}) {
    auto node = runtime::make_object<CandidateKernelNode>();
    node->rule_name_ = std::move(rule_name);
    node->priority_ = priority;
    node->sub_graph_ = std::move(sub_graph);
    node->spec_ = std::move(spec);
    node->function_ = std::move(function);
    // cost defaults to unknown
    data_ = std::move(node);
  }

  /*!
   * \brief Returns true if this and \p that candidate could be unioned. The result may not
   * be valid, but we at least check the respective sub-graphs are disjoint and at least one
   * output node of this is an entry node of that.
   */
  bool MaybeUnionable(const CandidateKernel& that) const;

  /*!
   * \brief Returns the disjoint union of this and \p that.
   */
  CandidateKernel DisjointUnion(const DataflowGraph& dataflow_graph,
                                const CandidateKernel& that) const;

  /*!
   * \brief Returns the disjoint union of all \p candidates.
   */
  static CandidateKernel DisjointUnion(const DataflowGraph& dataflow_graph,
                                       std::vector<CandidateKernel> candidates);

  TVM_DEFINE_OBJECT_REF_METHODS(CandidateKernel, ObjectRef, CandidateKernelNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_CANDIDATE_KERNEL_H_
