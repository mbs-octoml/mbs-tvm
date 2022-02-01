import torch
import argparse
import onnx
import onnxruntime
from resnets import resnet50
import torch.autograd.profiler as profiler
import tvm.relay.op
from tqdm import tqdm 
from tvm import relay
import tvm
from tvm import te
import numpy as np
import tvm.contrib.graph_executor as runtime
from tvm.relay import testing
from torchvision.models import resnet
import torch.nn as nn
from torch.nn.parameter import Parameter

torch.backends.cudnn.benchmark = True

NAME = 'bert'

D_MODEL = 1024
HEADS = 16

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.q = Parameter(data = torch.randn(D_MODEL, D_MODEL) * 0.01)
        self.k = Parameter(data = torch.randn(D_MODEL, D_MODEL) * 0.01)
        self.v = Parameter(data = torch.randn(D_MODEL, D_MODEL) * 0.01)
        
        self.relu = nn.ReLU()
        self.up = Parameter(data = torch.randn(D_MODEL, 4*D_MODEL) * 0.01)
        self.final = Parameter(data = torch.randn(4*D_MODEL, D_MODEL) * 0.01)
        

    def forward(self, x):
        q = torch.matmul(x, self.q)
        k = torch.matmul(x, self.k)
        v = torch.matmul(x, self.v)

        q = torch.reshape(q, (1,64,16,64)).permute(0,2,1,3)
        k = torch.reshape(k, (1,64,16,64)).permute(0,2,1,3)
        v = torch.reshape(v, (1,64,16,64)).permute(0,2,1,3)

        logits = torch.matmul(q, k)
        output = torch.matmul(logits, v)

        output = output.permute(0,2,1,3)
        output = torch.reshape(output, (1,64,1024))

        output = torch.matmul(output, self.up)
        output = self.relu(output)
        output = torch.matmul(output, self.final)
        return output

LAYERS = 8
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.nodes = nn.ModuleList([Attention() for i in range(LAYERS)])
    
    def forward(self, x):
        t = x
        for i in range(LAYERS):
            t = self.nodes[i](t)
        return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=5000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=1000)
    args = parser.parse_args()

    model = BERT().cuda()
    model.eval()
    inputs = torch.randn(1, 64, 1024).cuda()

    model(inputs)

    from torch2trt import torch2trt
    import time
    model_trt = torch2trt(model, [inputs])

    times = []
    for i in tqdm(range(args.discard_iter + args.iterations)):

        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        model_trt(inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        times.append(1000.0 * (t1 - t0))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("TensorRT: Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")