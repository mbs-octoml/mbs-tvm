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

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.batchnorm.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      DLTensor* x = args[1];
      DLTensor* y = args[2];
      DLTensor* scale = args[3];
      DLTensor* bias = args[4];
      DLTensor* mean = args[5];
      DLTensor* var = args[6];
      double double_alpha = args[7];
      double double_beta = args[8];
      double double_epsilon = args[9];
      const void* alpha;
      const void* beta;

      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->batchnorm_entry.mode = static_cast<cudnnBatchNormMode_t>(mode);
      entry_ptr->batchnorm_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->batchnorm_entry.data_type, double_alpha);
      beta = CuDNNDataType::GetConst(entry_ptr->batchnorm_entry.data_type, double_beta);

      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->batchnorm_entry.shape_desc, CUDNN_TENSOR_NCHW, 
        entry_ptr->batchnorm_entry.data_type,
        static_cast<int>(x->shape[0]), static_cast<int>(x->shape[1]),
        static_cast<int>(x->shape[2]), static_cast<int>(x->shape[3])));


      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->batchnorm_entry.scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, 
        entry_ptr->batchnorm_entry.data_type,
        static_cast<int>(mean->shape[0]), static_cast<int>(mean->shape[1]),
        static_cast<int>(mean->shape[2]), static_cast<int>(mean->shape[3])));


      CUDNN_CALL(cudnnBatchNormalizationForwardInference(entry_ptr->handle,
            entry_ptr->batchnorm_entry.mode,
            alpha,
            beta,
            entry_ptr->batchnorm_entry.shape_desc, 
            x->data,
            entry_ptr->batchnorm_entry.shape_desc, 
            y->data,
            entry_ptr->batchnorm_entry.scale_bias_mean_var_desc,
            scale->data,
            bias->data,
            mean->data,
            var->data,
            double_epsilon
            ));
    });

}  // namespace contrib
}  // namespace tvm
