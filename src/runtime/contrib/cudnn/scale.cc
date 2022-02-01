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
 * \file src/runtime/contrib/cudnn/softmax.cc
 * \brief Use external cudnn softmax function
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.scale")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* y = args[0];
      double double_alpha = args[1];
      const void* alpha;

      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->scale_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(y->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->scale_entry.data_type, double_alpha);

      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->scale_entry.shape_desc, CUDNN_TENSOR_NCHW, 
        entry_ptr->scale_entry.data_type,
        static_cast<int>(y->shape[0]), static_cast<int>(y->shape[1]),
        static_cast<int>(y->shape[2]), static_cast<int>(y->shape[3])));


      CUDNN_CALL(cudnnScaleTensor(entry_ptr->handle, 
            entry_ptr->scale_entry.shape_desc, 
            y->data, 
            alpha));
    });

}  // namespace contrib
}  // namespace tvm
