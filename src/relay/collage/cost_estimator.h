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
 * \file src/relay/collage/cost_estimator.cc
 * \brief Interface for measuring candidate kernel cost.
 */

#ifndef TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_
#define TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_

#include <tvm/relay/function.h>

#include "cost.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief An (abstract) estimator for the cost of executing a "Primitive" Relay \p Function
 * representing a candidate kernel, using a given target for lowering and codegen.
 *
 * Generally the implementation will compile to a \p runtime::Module (possibly on a target-specific
 * worker if cross-compilation is not available), repeatedly invoke the entry point with random
 * data until measure variance is acceptable (on a target-specific worker), and return the
 * summarized costs. The result may be cached, however the cache lookup and update is hidden.
 *
 * If using a TVM native \p Target, it is possible compilation will itself invoke TVM tuning. We
 * only care about the cost of the tuned kernel.
 *
 * TODO(mbs): Actually, currently not abstract so can get some local measurements.
 */
class CostEstimator {
 public:
  /*!
   * \brief Return the estimated cost (possibly after many many minutes of training time) of
   * running \p function using \p target.
   */
  virtual Cost Estimate(const Function& function, const Target& target) const;

  /*!
   * \brief As for \p Estimate, but use and update the internal in-memory cache.
   */
  Cost CachedEstimate(const Function& function, const Target& target);

 private:
  /*! \brief Returns string which is 1:1 with function and target. */
  std::string CacheKey(const Function& function, const Target& target);

  /*! \brief In-memory cache.
   * TODO(mbs): Just to get us going.
   */
  std::unordered_map<std::string, Cost> cache_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COST_ESTIMATOR_H_
