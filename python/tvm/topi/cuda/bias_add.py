# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
from tvm.target import Target
from tvm import te
from tvm.contrib import cudnn
from .. import generic
from .injective import schedule_injective


def schedule_bias_add(outs):
    """Schedule for bias_add op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return schedule_injective(outs)


def biasadd_cudnn(data, bias, axis=1):
    """Perform bias_add on the data using cudnn"""
    print("Python topi cuda cudnn bias_add!!")
    return cudnn.biasadd(data, bias, axis)


def schedule_biasadd_cudnn(outs):
    """Schedule for softmax cudnn op"""
    return generic.schedule_extern(outs)
