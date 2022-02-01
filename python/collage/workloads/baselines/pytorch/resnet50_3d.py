import torch
import argparse
import onnx
import onnxruntime
from resnets_3d import resnet50_3d
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

torch.backends.cudnn.benchmark = True

NAME = 'resnet50_3d'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=500)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=100)
    args = parser.parse_args()

    model = resnet50_3d().cuda()
    model.eval()
    inputs = torch.randn(1, 64, 3, 56, 56).cuda()

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

    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            model(inputs)
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations)
    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")

    input_shape = [1, 64, 3, 56, 56]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model.cpu(), input_data).eval()

    torch.jit.save(scripted_model, f'models/{NAME}.pth')

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    #print("Relay module function:\n", mod.astext(show_meta_data=True))

    with open(f"models/{NAME}.txt", "w") as text_file:
        text_file.write(mod.astext(show_meta_data=True))

    input_names = [ "input0" ]
    output_names = [ "output0" ]

    model.eval()

    with torch.no_grad():
        out_torch = model(inputs.cpu()).cpu().detach().numpy()

    torch.onnx.export(scripted_model, input_data, 
                    f"models/{NAME}.onnx", verbose=False, 
                    export_params=True,
                    do_constant_folding=False,
                    input_names=input_names, output_names=output_names, 
                    training = torch.onnx.TrainingMode.TRAINING,
                    example_outputs=torch.rand((1, 2048, 1, 7, 7)),
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
