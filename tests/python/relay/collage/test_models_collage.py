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

import logging
import os

from test_collage_fuse_ops import just_trt, describe_onnx, from_onnx, just_cutlass, run, just_tvm
from tpm import describe

MODEL_PATH = os.environ.get("ONNX_MODEL_ROOT")

RUN = True

models = {
    "mobilenet": [
        os.path.join(
            MODEL_PATH, "vision/classification/mobilenet/model/mobilenetv2-7.onnx"),
        {"input": [1, 3, 224, 224]},
        {"input": "float32"},
    ],
    "mnist": [
        os.path.join(
            MODEL_PATH, "vision/classification/mnist/model/mnist-1.onnx"),
        {"Input73": [1, 1, 28, 28]},
        {"Input73": "float32"},
    ],
    "squeezenet": [
        os.path.join(
            MODEL_PATH, "vision/classification/squeezenet/model/squeezenet1.0-12.onnx"),
        {"data_0": [1, 3, 224, 224]},
        {"data_0": "float32"},
    ],
    "vgg": [
        os.path.join(
            MODEL_PATH, "vision/classification/vgg/model/vgg19-7.onnx"),
        {"data": [1, 3, 224, 224]},
        {"data": "float32"},
    ],
    "resnet50-v1": [
        os.path.join(
            MODEL_PATH, "vision/classification/resnet/model/resnet50-v1-12.onnx"),
        {"data": [1, 3, 224, 224]},
        {"data": "float32"},
    ],
    "resnet18-v1.7": [
        os.path.join(
            MODEL_PATH, "vision/classification/resnet/model/resnet18-v1-7.onnx"),
        {"data": [1, 3, 224, 224]},
        {"data": "float32"},
    ],
    "efficientnet-lite": [
        os.path.join(
            MODEL_PATH, "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"
        ),
        {"images:0": [1, 224, 224, 3]},
        {"images:0": "float32"},
    ],
    "alexnet": [
        os.path.join(
            MODEL_PATH, "vision/classification/alexnet/model/bvlcalexnet-12.onnx"),
        {"data_0": [1, 3, 224, 224]},
        {"data_0": "float32"},
    ],
    "ssd": [
        os.path.join(
            MODEL_PATH, "vision/object_detection_segmentation/ssd/model/ssd-12.onnx"),
        {"image": [1, 3, 1200, 1200]},
        {"image": "float32"},
    ],
    "yolov3": [
        os.path.join(
            MODEL_PATH, "vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx"
        ),
        {"image": [1, 3, 1200, 1200]},
        {"image": "float32"},
    ],
    "mask-rcnn": [
        os.path.join(
            MODEL_PATH, "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx"
        ),
        {"image": [3, 1, 1]},
        {"image": "float32"},
    ],
    "gpt-2": [
        os.path.join(
            MODEL_PATH, "text/machine_comprehension/gpt-2/model/gpt2-10.onnx"),
        {"image": [3, 1, 1]},
        {"image": "float32"},
    ],
    "duc": [
        os.path.join(
            MODEL_PATH, "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx"
        ),
        {"data": [1, 3, 800, 800]},
        {"data": "float32"},
    ],
}


if __name__ == "__main__":

    # model_to_run = "efficientnet-lite"
    # model_to_run = "duc"
    model_to_run = "mobilenet"
    model, input_shapes, input_dtypes = models[model_to_run]

    if RUN:
        logging.info("*********************** Begin TVM run for " +
                     model_to_run + " ****")
        just_tvm(*from_onnx(model, input_shapes, input_dtypes))
        logging.info("*********************** Begin TRT run for " +
                     model_to_run + " ****")
        just_trt(*from_onnx(model, input_shapes, input_dtypes))
        logging.info("*********************** End TRT run for " +
                     model_to_run + " ****\n")

        logging.info("*********************** Begin CUTLASS run for " +
                     model_to_run + " ****")
        just_cutlass(*from_onnx(model, input_shapes, input_dtypes))
        logging.info("*********************** End CUTLASS run for " +
                     model_to_run + " ****\n")

        logging.info("*********************** Begin Collage run for " +
                     model_to_run + " ****")
        run(*from_onnx(model, input_shapes, input_dtypes))
        logging.info("*********************** End Collage run for " +
                     model_to_run + " ****\n")
    else:
        describe(model)
