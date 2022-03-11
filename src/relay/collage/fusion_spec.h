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
 * \file src/relay/collage/fusion_spec.h
 * \brief Combine a \p FusionRule with one or more \p Targets.
 */

#ifndef SRC_RELAY_COLLAGE_FUSION_SPEC_H_
#define SRC_RELAY_COLLAGE_FUSION_SPEC_H_

#include <tvm/relay/function.h>
#include <tvm/runtime/container/string.h>
#include <tvm/target/target.h>

#include "./fusion_rule.h"
#include "./sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

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
 * \brief Pairs a \p FusionRule with one or more \p Targets it can be used for. We also allow
 * each candidate kernel function to be rewritten before the candidate is used for estimating
 * kernel latency or included in the final 'partitioned' Relay expression.
 */
class FusionSpecNode : public Object {
 public:
  /*!
   * \brief Specification name to distinguish this spec from all others. Typically
   * the BYOC compiler name or "tvm".
   */
  String spec_name_;

  /*!
   * \brief The targets all candidate kernels should be compiled for. It is possible for multiple
   * target to share the same fusion rules and thus candidates, eg if we are targeting multiple
   * devices.
   */
  Array<Target> targets_;

  /*!
   * \brief The fusion rule to use to gather candidate kernels.
   */
  FusionRule rule_;

  /*!
   * \brief A function for processing the candidate kernel functions before handing off to
   * lowering/codegen.
   */
  TRewriteSubGraphFunc fused_result_func_ = DefaultRewriteSubGraphFunc;

  /*!
   * \brief Returns all the candidate kernels found by this fusion specification. The candidates
   * will be for a specific target, but will not yet have a function or cost.
   */
  Array<CandidateKernel> AllCandidateKernels(const DataflowGraph& dataflow_graph) const;

  std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.FusionSpec";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionSpecNode, Object);
};

class FusionSpec : public ObjectRef {
 public:
  FusionSpec(String spec_name, Array<Target> targets, FusionRule rule,
             TRewriteSubGraphFunc fused_result_func = DefaultRewriteSubGraphFunc);

  TVM_DEFINE_OBJECT_REF_METHODS(FusionSpec, ObjectRef, FusionSpecNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_FUSION_SPEC_H_
