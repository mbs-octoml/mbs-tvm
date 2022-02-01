from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from collections import OrderedDict

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # We may need inplace operators because copy above causes an error when translating it into TVM IR.
    # i.e., aten::copy_ is not supported by TVM
    # torch.sigmoid_(prediction[:, :, 0])
    # torch.sigmoid_(prediction[:, :, 1])
    # torch.sigmoid_(prediction[:, :, 4])

    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    # if True:
    #     x_offset = x_offset.cuda()
    #     y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    # if True:
    #     anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def get_test_input():
    img = cv2.imread("imgs/img2.jpg")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YoloV3(nn.Module):

    def __init__(self):
        super(YoloV3, self).__init__()
        self.image_size = 416
        self.module_list = nn.ModuleList()
        self.blocks = []

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_0", nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
                    ("batch_norm_0", nn.BatchNorm2d(32)),
                    ("leaky_0", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type':'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_1", nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
                    ("batch_norm_1", nn.BatchNorm2d(64)),
                    ("leaky_1", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_2", nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    ("batch_norm_2", nn.BatchNorm2d(32)),
                    ("leaky_2", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_3", nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_3", nn.BatchNorm2d(64)),
                    ("leaky_3", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_4", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_5", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)),
                    ("batch_norm_5", nn.BatchNorm2d(128)),
                    ("leaky_5", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_6", nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_6", nn.BatchNorm2d(64)),
                    ("leaky_6", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_7", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_7", nn.BatchNorm2d(128)),
                    ("leaky_7", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_8", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_9", nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_9", nn.BatchNorm2d(64)),
                    ("leaky_9", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_10", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_10", nn.BatchNorm2d(128)),
                    ("leaky_10", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_11", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_12", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)),
                    ("batch_norm_12", nn.BatchNorm2d(256)),
                    ("leaky_12", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_13", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_13", nn.BatchNorm2d(128)),
                    ("leaky_13", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_14", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_14", nn.BatchNorm2d(256)),
                    ("leaky_14", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_15", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_16", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_16", nn.BatchNorm2d(128)),
                    ("leaky_16", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_17", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_17", nn.BatchNorm2d(256)),
                    ("leaky_17", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_18", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_19", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_19", nn.BatchNorm2d(128)),
                    ("leaky_19", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_20", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_20", nn.BatchNorm2d(256)),
                    ("leaky_20", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_21", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_22", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_22", nn.BatchNorm2d(128)),
                    ("leaky_22", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_23", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_23", nn.BatchNorm2d(256)),
                    ("leaky_23", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_24", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_25", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_25", nn.BatchNorm2d(128)),
                    ("leaky_25", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_26", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_26", nn.BatchNorm2d(256)),
                    ("leaky_26", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_27", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_28", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_28", nn.BatchNorm2d(128)),
                    ("leaky_28", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_29", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_29", nn.BatchNorm2d(256)),
                    ("leaky_29", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_30", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_31", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_31", nn.BatchNorm2d(128)),
                    ("leaky_31", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_32", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_32", nn.BatchNorm2d(256)),
                    ("leaky_32", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_33", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_34", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_34", nn.BatchNorm2d(128)),
                    ("leaky_34", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_35", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_35", nn.BatchNorm2d(256)),
                    ("leaky_35", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_36", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_37", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)),
                    ("batch_norm_37", nn.BatchNorm2d(512)),
                    ("leaky_37", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_38", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_38", nn.BatchNorm2d(256)),
                    ("leaky_38", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_39", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_39", nn.BatchNorm2d(512)),
                    ("leaky_39", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_40", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_41", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_41", nn.BatchNorm2d(256)),
                    ("leaky_41", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_42", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_42", nn.BatchNorm2d(512)),
                    ("leaky_42", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_43", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_44", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_44", nn.BatchNorm2d(256)),
                    ("leaky_44", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_45", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_45", nn.BatchNorm2d(512)),
                    ("leaky_45", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_46", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_47", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_47", nn.BatchNorm2d(256)),
                    ("leaky_47", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_48", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_48", nn.BatchNorm2d(512)),
                    ("leaky_48", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_49", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_50", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_50", nn.BatchNorm2d(256)),
                    ("leaky_50", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_51", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_51", nn.BatchNorm2d(512)),
                    ("leaky_51", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_52", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_53", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_53", nn.BatchNorm2d(256)),
                    ("leaky_53", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_54", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_54", nn.BatchNorm2d(512)),
                    ("leaky_54", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_55", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_56", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_56", nn.BatchNorm2d(256)),
                    ("leaky_56", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_57", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_57", nn.BatchNorm2d(512)),
                    ("leaky_57", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_58", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_59", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_59", nn.BatchNorm2d(256)),
                    ("leaky_59", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_60", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_60", nn.BatchNorm2d(512)),
                    ("leaky_60", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_61", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_62", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)),
                    ("batch_norm_62", nn.BatchNorm2d(1024)),
                    ("leaky_62", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_63", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_63", nn.BatchNorm2d(512)),
                    ("leaky_63", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_64", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_64", nn.BatchNorm2d(1024)),
                    ("leaky_64", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_65", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_66", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_66", nn.BatchNorm2d(512)),
                    ("leaky_66", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_67", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_67", nn.BatchNorm2d(1024)),
                    ("leaky_67", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_68", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_69", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_69", nn.BatchNorm2d(512)),
                    ("leaky_69", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_70", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_70", nn.BatchNorm2d(1024)),
                    ("leaky_70", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_71", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_72", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_72", nn.BatchNorm2d(512)),
                    ("leaky_72", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_73", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_73", nn.BatchNorm2d(1024)),
                    ("leaky_73", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("shortcut_74", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'shortcut', 'from': '-3'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_75", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_75", nn.BatchNorm2d(512)),
                    ("leaky_75", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_76", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_76", nn.BatchNorm2d(1024)),
                    ("leaky_76", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_77", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_77", nn.BatchNorm2d(512)),
                    ("leaky_77", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_78", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_78", nn.BatchNorm2d(1024)),
                    ("leaky_78", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_79", nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_79", nn.BatchNorm2d(512)),
                    ("leaky_79", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_80", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_80", nn.BatchNorm2d(1024)),
                    ("leaky_80", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_81", nn.Conv2d(1024, 75, kernel_size=(1, 1), stride=(1, 1)))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("Dectection_82", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'yolo', 'anchors': '116,90,  156,198,  373,326'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("route_83", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'route', 'layers': '-4'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_84", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_84", nn.BatchNorm2d(256)),
                    ("leaky_84", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("upsample_85", nn.Upsample(scale_factor=2, mode='nearest')),
                    ])))
        self.blocks.append({'type': 'upsample'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("route_86", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'route', 'layers': '-1, 61'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_87", nn.Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_87", nn.BatchNorm2d(256)),
                    ("leaky_87", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_88", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_88", nn.BatchNorm2d(512)),
                    ("leaky_88", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_89", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_89", nn.BatchNorm2d(256)),
                    ("leaky_89", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_90", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_90", nn.BatchNorm2d(512)),
                    ("leaky_90", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_91", nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_91", nn.BatchNorm2d(256)),
                    ("leaky_91", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_92", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_92", nn.BatchNorm2d(512)),
                    ("leaky_92", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_93", nn.Conv2d(512, 75, kernel_size=(1, 1), stride=(1, 1)))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("Dectection_94", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'yolo', 'anchors': '30,61,  62,45,  59,119'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("route_95", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'route', 'layers': '-4'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_96", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_96", nn.BatchNorm2d(128)),
                    ("leaky_96", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("upsample_97", nn.Upsample(scale_factor=2, mode='nearest')),
                    ])))
        self.blocks.append({'type': 'upsample'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("route_98", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'route', 'layers': '-1, 36'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_99", nn.Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_99", nn.BatchNorm2d(128)),
                    ("leaky_99", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_100", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_100", nn.BatchNorm2d(256)),
                    ("leaky_100", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_101", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_101", nn.BatchNorm2d(128)),
                    ("leaky_101", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_102", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_102", nn.BatchNorm2d(256)),
                    ("leaky_102", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_103", nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1),bias=False)),
                    ("batch_norm_103", nn.BatchNorm2d(128)),
                    ("leaky_103", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_104", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),bias=False)),
                    ("batch_norm_104", nn.BatchNorm2d(256)),
                    ("leaky_104", nn.ReLU(inplace=True))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("conv_105", nn.Conv2d(256, 75, kernel_size=(1, 1), stride=(1, 1)))
                    ])))
        self.blocks.append({'type': 'convolutional'})

        self.module_list.append(nn.Sequential(OrderedDict([
                    ("Dectection_106", EmptyLayer()),
                    ])))
        self.blocks.append({'type': 'yolo', 'anchors': '10,13,  16,30,  33,23'})


    def forward(self, x):

        modules = self.blocks
        outputs = {}  # cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"].split(",")
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':

                anchors = module["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                # Get the input dimensions
                inp_dim = int(self.image_size)
                # Get the number of classes
                num_classes = 20

                # Transform
                # x = x.data
                # print("x-data", x.shape)
                # x = predict_transform(x, inp_dim, anchors, num_classes, True)
                # if not write:  # if no collector has been intialised.
                #     detections = x
                #     write = 1
                #
                # else:
                #     detections = torch.cat((detections, x), 1)
                # print("transformed", x.shape)

            outputs[i] = x
        return x
        # return detections#, outputs[105], outputs[93], outputs[81]



if __name__ == '__main__':

    model = YoloV3()
    print(model)
    # inp = get_test_input()
    inp = torch.randn((1,3,416,416))
    pred = model(inp)
    # pred = model(inp, False)
    # pred, _, _, _ = model(inp, False)
    print(pred.shape)
