import torch
import argparse
import onnx
import onnxruntime
from .resnets import resnet50
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
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor 

torch.backends.cudnn.benchmark = True

NAME = 'nasneta'

class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

class NormalCell(nn.Module):
    def __init__(self, channels_cur, channels_prev, out_channels):
        super(NormalCell, self).__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(channels_cur, out_channels, 1, stride=1, padding=0), nn.ReLU())
        self.fit = nn.Sequential(nn.Conv2d(channels_prev, out_channels, 1, stride=1, padding=0), nn.ReLU())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.nodes = nn.ModuleList([SeparableConv2d(out_channels, out_channels, 3, 1, 1) for i in range(5)])

    def forward(self, prev, cur):
        cur = self.squeeze(cur)
        prev = self.fit(prev)

        ts = []
        ts.append(self.nodes[0](cur))
        ts.append(cur)
        ts.append(self.nodes[1](prev))
        ts.append(self.nodes[2](cur))
        ts.append(self.pool(cur))
        ts.append(prev)
        ts.append(self.pool(prev))
        ts.append(self.pool(prev))
        ts.append(self.nodes[3](prev))
        ts.append(self.nodes[4](prev))

        assert len(ts) == 10
        outputs = []
        for i in range(5):
            outputs.append(torch.add(ts[2*i], ts[2*i+1]))
        return torch.cat(outputs, dim=1)

class ReductionCell(nn.Module):
    def __init__(self, channels_cur, channels_prev, out_channels):
        super(ReductionCell, self).__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(channels_cur, out_channels, 1, stride=1, padding=0), nn.ReLU())
        self.fit = nn.Sequential(nn.Conv2d(channels_prev, out_channels, 1, stride=1, padding=0), nn.ReLU())
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.avgpool_1 = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_channels = out_channels
        self.conv7_1 = SeparableConv2d(out_channels, out_channels, 7, 2, 3)
        self.conv7_2 = SeparableConv2d(out_channels, out_channels, 7, 2, 3)
        self.conv5_1 = SeparableConv2d(out_channels, out_channels, 5, 2, 2)
        self.conv5_2 = SeparableConv2d(out_channels, out_channels, 5, 2, 2)
        self.conv3_1 = SeparableConv2d(out_channels, out_channels, 3, 1, 1)
        
    def forward(self, prev, cur):

        cur = self.squeeze(cur)
        prev = self.fit(prev)

        ts = []
        outputs = []

        ts.append(self.conv7_1(prev))
        ts.append(self.conv5_1(cur))
        outputs.append(torch.add(ts[0], ts[1]))
        ts.append(self.maxpool(cur))
        ts.append(self.conv7_2(prev))
        outputs.append(torch.add(ts[2], ts[3]))
        ts.append(self.avgpool(cur))
        ts.append(self.conv5_2(prev))
        outputs.append(torch.add(ts[4], ts[5]))
        ts.append(self.maxpool(cur))
        ts.append(self.conv3_1(outputs[0]))
        outputs.append(torch.add(ts[6], ts[7]))
        ts.append(self.avgpool_1(outputs[0]))
        ts.append(outputs[1])
        outputs.append(torch.add(ts[8], ts[9]))
        return torch.cat(outputs, dim=1)

REPEAT = 5
CHANNEL = 64

class NASNetA(nn.Module):
    def __init__(self):
        super(NASNetA, self).__init__()
        first_cell = [NormalCell(CHANNEL, CHANNEL, CHANNEL)] + [NormalCell(CHANNEL*5, CHANNEL, CHANNEL)] + [NormalCell(CHANNEL*5, CHANNEL*5, CHANNEL) for i in range(REPEAT - 2)]
        self.normal_cell_0 = nn.ModuleList(first_cell)
        self.normal_cell_1 = nn.ModuleList([NormalCell(CHANNEL*10, CHANNEL*10, CHANNEL*2) for i in range(REPEAT)])
        self.normal_cell_2 = nn.ModuleList([NormalCell(CHANNEL*20, CHANNEL*20, CHANNEL*4) for i in range(REPEAT)])
        self.reduction_cell_0 = ReductionCell(CHANNEL*5, CHANNEL*5, CHANNEL*2)
        self.reduction_cell_1 = ReductionCell(CHANNEL*10, CHANNEL*10, CHANNEL*4)
        
    def forward(self, x):
        input = x
        out_channels = CHANNEL
    
        prev = input
        cur = input
        for j in range(REPEAT):
            t = self.normal_cell_0[j](prev, cur)
            prev = cur
            cur = t
        out_channels *= 2
        
        
        input = self.reduction_cell_0(prev, cur)
        prev = input
        cur = input
        for j in range(REPEAT):
            t = self.normal_cell_1[j](prev, cur)
            prev = cur
            cur = t
        out_channels *= 2

        input = self.reduction_cell_1(prev, cur)
        prev = input
        cur = input
        for j in range(REPEAT):
            t = self.normal_cell_2[j](prev, cur)
            prev = cur
            cur = t
        out_channels *= 2

        return cur

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=5000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=1000)
    args = parser.parse_args()

    model = NASNetA().cuda()
    inputs = torch.randn(1,CHANNEL,56,56).cuda()

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

    print(model(inputs).size())

    scripted_model = torch.jit.trace(model, inputs)

    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            scripted_model(inputs)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    input_shape = [1,CHANNEL,56,56]
    input_data = torch.randn(input_shape)

    torch.jit.save(scripted_model, f'models/{NAME}.pth')

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    with open(f"models/{NAME}.txt", "w") as text_file:
        text_file.write(mod.astext(show_meta_data=True))

    input_names = [ "input0" ]
    output_names = [ "output0" ]

    model.eval()
    model.cpu()


    with torch.no_grad():
        out_torch = model(inputs.cpu()).cpu().detach().numpy()

    torch.onnx.export(scripted_model, input_data,
                      f"models/{NAME}.onnx", verbose=False,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      training = torch.onnx.TrainingMode.EVAL,
                      example_outputs=torch.randn((1,CHANNEL*20,14,14)),
                      opset_version=12)

    onnx_model = onnx.load(f"models/{NAME}.onnx")

    sess = onnxruntime.InferenceSession(f"models/{NAME}.onnx")
    out_onnx = sess.run(["output0"], {"input0": inputs.cpu().numpy()})[0]

    input_name = "input0"
    shape_dict = {input_name: input_shape}
    mod2, params2 = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    with open(f"models/{NAME}_onnx.txt", "w") as text_file:
        text_file.write(mod2.astext(show_meta_data=True))

    # Bulid the subgraph
    ctx = tvm.device("cuda", 0)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", target_host="llvm", params=params)

    with tvm.transform.PassContext(opt_level=3):
        lib2 = relay.build(mod2, target="cuda", target_host="llvm", params=params2)

    m = runtime.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))

    m2 = runtime.GraphModule(lib2["default"](ctx))
    # Set inputs
    m2.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))

    # Measure performance
    ftimer = m.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    perf = np.mean(prof_res)
    print("%.5f ms" % (perf))

    ftimer = m2.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    perf = np.mean(prof_res)
    print("%.5f ms" % (perf))

    m.run()
    out = m.get_output(0)
    out_tvm = out.asnumpy()

    m2.run()
    out = m2.get_output(0)
    out_tvm2 = out.asnumpy()

    print(out_tvm[0,:10,0,0])
    print(out_tvm2[0,:10,0,0])
    print(out_torch[0,:10,0,0])
    print(out_onnx[0,:10,0,0])

    TOL = 1e-01
    assert np.allclose(out_onnx, out_torch, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm, rtol=TOL, atol=TOL)
    assert np.allclose(out_onnx, out_tvm2, rtol=TOL, atol=TOL)
    assert np.allclose(out_torch, out_tvm2, rtol=TOL, atol=TOL)

    print(np.abs((out_torch - out_tvm)).max())