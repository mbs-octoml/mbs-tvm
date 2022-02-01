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
 * \file Use external cudnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_

#include <cudnn.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include <string>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {

#define CUDNN_CALL(func)                                                       \
  {                                                                            \
    cudnnStatus_t e = (func);                                                  \
    ICHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

/*! breif Convert DLTensor type to CuDNN type */
struct CuDNNDataType {
  static cudnnDataType_t DLTypeToCuDNNType(const DLDataType& dtype);
  template <int v>
  static const void* GetConst(cudnnDataType_t type);
  static const void* GetConst(cudnnDataType_t type, double val);
};  // struct CuDNNDataType

inline void GetStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    mul *= dims[i];
    strides[i] = mul;
  }
}

inline void GetCudnnStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    strides[i] = mul;
    mul *= dims[i];
  }
}

struct FusedOpsEntry {
  cudnnFusedOpsPlan_t fuse_plan;
  cudnnFusedOpsVariantParamPack_t varPack;
  cudnnFusedOpsConstParamPack_t constPack;
  FusedOpsEntry();
  ~FusedOpsEntry();
};  // FusedOpsEntry


struct FusedConvEntry {
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionMode_t mode;
  cudnnFilterDescriptor_t filter_desc;
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t z_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnActivationDescriptor_t activation_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  // cudnnMathType_t math_type;
  //TVMContext ctx;
  Device device;
  //DLContext ctx;
  runtime::DeviceAPI* cuda_api;
  void* workspace{nullptr};
  size_t workspace_size{0};
  FusedConvEntry();
  ~FusedConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // FusedConvThreadEntry


struct FusedMatmulEntry {
  //cudnnConvolutionDescriptor_t conv_desc;
  //cudnnConvolutionMode_t mode;
  //cudnnFilterDescriptor_t filter_desc;
  //cudnnConvolutionFwdAlgo_t fwd_algo;

  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnActivationDescriptor_t activation_desc;
  // cudnnMathType_t math_type;
  //TVMContext ctx;
  Device device;
  //DLContext ctx;
  runtime::DeviceAPI* cuda_api;
  void* workspace{nullptr};
  size_t workspace_size{0};
  FusedMatmulEntry();
  ~FusedMatmulEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // FusedMatmulThreadEntry

struct ConvEntry {
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionMode_t mode{CUDNN_CROSS_CORRELATION};
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  // cudnnMathType_t math_type;
  Device device;
  runtime::DeviceAPI* cuda_api;
  void* workspace{nullptr};
  size_t workspace_size{0};
  ConvEntry();
  ~ConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // ConvThreadEntry

struct SoftmaxEntry {
  cudnnSoftmaxMode_t mode;
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  SoftmaxEntry();
  ~SoftmaxEntry();
};  // SoftmaxEntry

struct BiasEntry {
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  BiasEntry();
  ~BiasEntry();
}; // BiasEntry

struct BatchNormEntry {
  cudnnDataType_t data_type;
  cudnnBatchNormMode_t mode;
  cudnnTensorDescriptor_t shape_desc;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc;
  BatchNormEntry();
  ~BatchNormEntry();
}; // BatchNormEntry

struct PoolingEntry {
  cudnnDataType_t data_type;
  cudnnPoolingDescriptor_t pooling_desc;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  PoolingEntry();
  ~PoolingEntry();
}; // PoolingEntry

struct ScaleEntry {
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  ScaleEntry();
  ~ScaleEntry();
}; // ScaleEntry

struct ReduceEntry {
  cudnnDataType_t data_type;
  cudnnReduceTensorDescriptor_t reduce_desc;
  cudnnTensorDescriptor_t a_desc;
  cudnnTensorDescriptor_t c_desc;
  //TVMContext ctx;
  Device device;
  runtime::DeviceAPI* cuda_api;
  void* workspace{nullptr};
  size_t workspace_size{0};
  ReduceEntry();
  ~ReduceEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
}; // ReduceEntry

struct ActivationEntry {
  cudnnActivationDescriptor_t activation_desc;
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  ActivationEntry();
  ~ActivationEntry();
}; // ActivationEntry

struct CuDNNThreadEntry {
  CuDNNThreadEntry();
  ~CuDNNThreadEntry();

  bool exists() const { return handle; }

  cudnnHandle_t handle{nullptr};
  ConvEntry conv_entry;
  SoftmaxEntry softmax_entry;
  FusedConvEntry fused_conv_entry;
  FusedMatmulEntry fused_matmul_entry;
  BiasEntry bias_entry;
  BatchNormEntry batchnorm_entry;
  PoolingEntry pooling_entry;
  ScaleEntry scale_entry;
  ReduceEntry reduce_entry;
  FusedOpsEntry fused_ops_entry;
  ActivationEntry activation_entry;
  runtime::DeviceAPI* cuda_api{nullptr};
  static CuDNNThreadEntry* ThreadLocal(bool check_exists = true);
};  // CuDNNThreadEntry

void SetConvDescriptors(CuDNNThreadEntry* entry_ptr, int format, int dims, int groups,
                        const int pad[], const int stride[], const int dilation[], int64_t x_dim[],
                        int64_t w_dim[], int64_t y_dim[], DLDataType data_dtype,
                        const std::string& conv_dtype);

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
