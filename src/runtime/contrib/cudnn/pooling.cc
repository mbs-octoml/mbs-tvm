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

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.pooling.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* x = args[0];
      DLTensor* y = args[1];
      double double_alpha = args[2];
      double double_beta = args[3];
      int mode = args[4];
      int nanOpt = args[5];
      int windowHeight = args[6];
      int windowWidth = args[7];
      int verticalPadding = args[8];
      int horizontalPadding = args[9];
      int verticalStride = args[10];
      int horizontalStride = args[11];
      const void* alpha;
      const void* beta;
      
      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->pooling_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->pooling_entry.data_type, double_alpha);
      beta = CuDNNDataType::GetConst(entry_ptr->pooling_entry.data_type, double_beta);

      // Set Pooling desc
      CUDNN_CALL(cudnnSetPooling2dDescriptor(entry_ptr->pooling_entry.pooling_desc,
        static_cast<cudnnPoolingMode_t>(mode),
        static_cast<cudnnNanPropagation_t>(nanOpt),
        windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride
        ));

      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->pooling_entry.input_desc, CUDNN_TENSOR_NCHW, 
        entry_ptr->pooling_entry.data_type,
        static_cast<int>(x->shape[0]), static_cast<int>(x->shape[1]),
        static_cast<int>(x->shape[2]), static_cast<int>(x->shape[3])));


      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->pooling_entry.output_desc, CUDNN_TENSOR_NCHW, 
        entry_ptr->pooling_entry.data_type,
        static_cast<int>(y->shape[0]), static_cast<int>(y->shape[1]),
        static_cast<int>(y->shape[2]), static_cast<int>(y->shape[3])));


      CUDNN_CALL(cudnnPoolingForward(entry_ptr->handle, 
            entry_ptr->pooling_entry.pooling_desc, 
            alpha,
            entry_ptr->pooling_entry.input_desc, 
            x->data, 
            beta,
            entry_ptr->pooling_entry.output_desc, 
            y->data));
    });

}  // namespace contrib
}  // namespace tvm
