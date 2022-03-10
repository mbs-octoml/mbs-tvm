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
 * \file src/relay/collage/utils.h
 * \brief Misc helpers.
 */

#ifndef SRC_RELAY_COLLAGE_UTILS_H_
#define SRC_RELAY_COLLAGE_UTILS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/container/string.h>

namespace tvm {
namespace relay {
namespace collage {

/*! \brief Returns \p "<left>+<right>". */
String UnionLabels(String left, String right);

/*! \brief Returns \p "<outer>.<inner>". */
String NestLabels(String outer, String inner);

/*! \brief Returns abbreviation for \p kind. */
std::string KindToString(OpPatternKind kind);

/*! \brief Returns maximum of \p left and \p right. */
OpPatternKind CombineKinds(OpPatternKind left, OpPatternKind right);

/*!
 * \brief Returns true if \p constant_node is a float/int/bool scalar which is always safe to
 * inline.
 */
bool IsSimpleScalar(const ConstantNode* constant_node);

/*!
 * \brief Returns true if \p expr can be safely inlined in body of function extracted
 * from sub-graph, even if \p expr was not technically matched by the pattern which produced
 * the sub-graph.
 */
bool CanInline(const Expr& expr);

/*!
 * \brief Returns true if \p op_node can be directly handled by the VM.
 */
bool IsSpecialOp(const OpNode* op_node);

/*!
 * \brief Return true if the Relay expression node given by \p expr cannot be evaluated by
 * the VM and must end up in a kernel.
 */
bool MustBeLowered(const Expr& expr);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // SRC_RELAY_COLLAGE_UTILS_H_
