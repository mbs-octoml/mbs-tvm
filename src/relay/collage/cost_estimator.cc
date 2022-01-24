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

#include "./cost_estimator.h"

namespace tvm {
namespace relay {
namespace collage {

Cost CostEstimator::Estimate(const Function& function, const Target& target) const {
  static const runtime::PackedFunc* estimate_seconds =
      runtime::Registry::Get("tvm.relay.collage.estimate_seconds");
  ICHECK(estimate_seconds);
  const double value = (*estimate_seconds)(function, target);
  if (std::isinf(value)) {
    return Cost::Invalid();
  } else if (std::isnan(value)) {
    return Cost::Unknown();
  } else {
    return Cost::Value(value);
  }
}

Cost CostEstimator::CachedEstimate(const Function& function, const Target& target) {
  std::string key = CacheKey(function, target);
  auto itr = cache_.find(key);
  if (itr != cache_.end()) {
    VLOG(1) << "Reusing cached candidate kernel cost";
    return itr->second;
  }
  Cost cost = Estimate(function, target);
  cache_.emplace(key, cost);
  return cost;
}

std::string CostEstimator::CacheKey(const Function& function, const Target& target) {
  std::ostringstream os;
  os << "{";
  os << PrettyPrint(function);
  os << "}{";
  os << target->ToDebugString();
  os << "}";
  return os.str();
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm