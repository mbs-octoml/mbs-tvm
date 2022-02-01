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

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.activation.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* x = args[0];
      DLTensor* y = args[1];
      double double_alpha = args[2];
      double double_beta = args[3];
      const void* alpha;
      const void* beta;
      int mode = args[4];
      int nanOpt = args[5];
      double coeff = args[6];

      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->activation_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->activation_entry.data_type, double_alpha);
      beta = CuDNNDataType::GetConst(entry_ptr->activation_entry.data_type, double_beta);

      // Set Activation
      CUDNN_CALL(cudnnSetActivationDescriptor(entry_ptr->activation_entry.activation_desc,
        static_cast<cudnnActivationMode_t>(mode),
        static_cast<cudnnNanPropagation_t>(nanOpt),
        coeff
        ));

      int ndim = x->ndim;
      // cuDNN only supports 4d or 5d
      ICHECK(ndim == 4 || ndim == 5);

      if(ndim == 4){
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
          entry_ptr->activation_entry.shape_desc, CUDNN_TENSOR_NCHW, 
          entry_ptr->activation_entry.data_type,
          static_cast<int>(x->shape[0]), static_cast<int>(x->shape[1]),
          static_cast<int>(x->shape[2]), static_cast<int>(x->shape[3])));
      }else{
        std::vector<int> dims(ndim);
        std::vector<int> tensor_stride(ndim);
        // Set Filter
        for (int i = 0; i < ndim; i++) {
          dims[i] = static_cast<int>(x->shape[i]);
        }
        GetCudnnStride(ndim, dims.data(), tensor_stride.data());
        CUDNN_CALL(cudnnSetTensorNdDescriptor(
              entry_ptr->activation_entry.shape_desc, 
              entry_ptr->activation_entry.data_type,     
              ndim, dims.data(),
              tensor_stride.data()));
      }


      CUDNN_CALL(cudnnActivationForward(entry_ptr->handle, 
            entry_ptr->activation_entry.activation_desc, 
            alpha,
            entry_ptr->activation_entry.shape_desc, 
            x->data, 
            beta,
            entry_ptr->activation_entry.shape_desc, 
            y->data));
    });

}  // namespace contrib
}  // namespace tvm
