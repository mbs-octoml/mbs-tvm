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

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.add")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* x = args[0];
      DLTensor* y = args[1];
      double double_alpha = args[2];
      double double_beta = args[3];
      int axis = args[4];

      int ndim = x->ndim;
      std::cerr << "Cudnn Add\n";
      ICHECK(ndim==2);

      std::cerr << "dim == 2\n";
      int64_t* shape = x->shape;
      if (axis < 0) axis += ndim;
      ICHECK(axis >= 0 && axis < ndim);
      const void* alpha;
      const void* beta;

      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->bias_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->bias_entry.data_type, double_alpha);
      beta = CuDNNDataType::GetConst(entry_ptr->bias_entry.data_type, double_beta);

      std::cerr << "Create Descriptors\n";
      // Set mode and shape descriptor
      if (axis == ndim - 1) {
        int64_t N = 1;
        for (int i = 0; i < ndim - 1; ++i) {
          N *= shape[i];
        }
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
          entry_ptr->bias_entry.shape_desc, CUDNN_TENSOR_NCHW, 
          entry_ptr->bias_entry.data_type,
          static_cast<int>(N),
          static_cast<int>(shape[ndim - 1]), 1, 1));
      }else{
        int64_t pre_axis_dim = 1;
        int64_t post_axis_dim = 1;
        for (int i = 0; i < ndim; ++i) {
          if (i < axis) {
            pre_axis_dim *= shape[i];
          } else if (i > axis) {
            post_axis_dim *= shape[i];
          }
        }
        CUDNN_CALL(cudnnSetTensor4dDescriptor(
          entry_ptr->bias_entry.shape_desc, CUDNN_TENSOR_NCHW, 
          entry_ptr->bias_entry.data_type,
            static_cast<int>(pre_axis_dim),
            static_cast<int>(shape[axis]), static_cast<int>(post_axis_dim), 1));

      }

      std::cerr << "Run\n";
      CUDNN_CALL(cudnnAddTensor(entry_ptr->handle, 
            alpha,
            entry_ptr->bias_entry.shape_desc, 
            x->data, 
            beta,
            entry_ptr->bias_entry.shape_desc, 
            y->data));
    });

}  // namespace contrib
}  // namespace tvm
