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

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.reduce")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* x = args[0];
      DLTensor* y = args[1];
      double double_alpha = args[2];
      double double_beta = args[3];
      const void* alpha;
      const void* beta;
      int mode = args[4];
      int nanOpt = args[5];
      int indiceOpt = args[6];
      int indiceType = args[7];
      void* indices = args[8]; // if there is a bug, suspect this first


      CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
      entry_ptr->reduce_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

      alpha = CuDNNDataType::GetConst(entry_ptr->reduce_entry.data_type, double_alpha);
      beta = CuDNNDataType::GetConst(entry_ptr->reduce_entry.data_type, double_beta);

      // reduce desc
      CUDNN_CALL(cudnnSetReduceTensorDescriptor(
            entry_ptr->reduce_entry.reduce_desc,
            static_cast<cudnnReduceTensorOp_t>(mode),
            entry_ptr->reduce_entry.data_type,
            static_cast<cudnnNanPropagation_t>(nanOpt),
            static_cast<cudnnReduceTensorIndices_t>(indiceOpt),
            static_cast<cudnnIndicesType_t>(indiceType)
            ));

      // input tensor desc
      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->reduce_entry.a_desc, CUDNN_TENSOR_NCHW,
        entry_ptr->reduce_entry.data_type,
        static_cast<int>(x->shape[0]), static_cast<int>(x->shape[1]),
        static_cast<int>(x->shape[2]), static_cast<int>(x->shape[3])));

      // output tensor desc
      CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->reduce_entry.c_desc, CUDNN_TENSOR_NCHW,
        entry_ptr->reduce_entry.data_type,
        static_cast<int>(y->shape[0]), static_cast<int>(y->shape[1]),
        static_cast<int>(y->shape[2]), static_cast<int>(y->shape[3])));

      size_t indices_size = 0;
      CUDNN_CALL(cudnnGetReductionIndicesSize(entry_ptr->handle, entry_ptr->reduce_entry.reduce_desc,
            entry_ptr->reduce_entry.a_desc, entry_ptr->reduce_entry.c_desc, &indices_size));

      // Set workspace
      size_t workspace_size = 0;
      CUDNN_CALL(cudnnGetReductionWorkspaceSize(
            entry_ptr->handle, entry_ptr->reduce_entry.reduce_desc,
            entry_ptr->reduce_entry.a_desc,
            entry_ptr->reduce_entry.c_desc,
            &workspace_size));
      entry_ptr->reduce_entry.UpdateWorkspace(workspace_size);


      CUDNN_CALL(cudnnReduceTensor(entry_ptr->handle,
            entry_ptr->reduce_entry.reduce_desc,
            indices, // NOTE: This may be an issue
            indices_size,
            entry_ptr->reduce_entry.workspace,
            workspace_size,
            alpha,
            entry_ptr->reduce_entry.a_desc,
            x->data,
            beta,
            entry_ptr->reduce_entry.c_desc,
            y->data));
    });

}  // namespace contrib
}  // namespace tvm
