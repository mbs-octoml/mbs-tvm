import argparse
import onnx
import onnxruntime
import tvm.relay.op
from tqdm import tqdm 
from tvm import relay
import tvm
from tvm import te
import numpy as np
import tvm.contrib.graph_executor as runtime
from tvm.relay import testing

models = [('bert', [64,1024], np.random.random_sample((64,1024))),
          ('nasneta', [1,64,56,56], np.random.random_sample((1,64,56,56))),
          ('nasrnn', [1,512], np.random.random_sample((1,512))),
          ('resnet50', [1,64,56,56], np.random.random_sample((1,64,56,56))),
          ('resnext50', [1,64,56,56], np.random.random_sample((1,64,56,56))),
          ('resnet50_3d', [1,64,3,56,56], np.random.random_sample((1,64,3,56,56))),
          ('mobilenet_v2', [1,32,56,56], np.random.random_sample((1,32,56,56)))
         ]

result_dict = {}

for NAME, input_shape, inputs in models:

    model = onnx.load(f"models/{NAME}_taso.onnx")

    output = [node.name for node in model.graph.output]
    input_all = [node.name for node in model.graph.input]
    input_initializer =  [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))

    print(net_feed_input, output)
    #sess = onnxruntime.InferenceSession(f"models/{NAME}_taso.onnx")
    #out_onnx = sess.run(["output"], {"data": inputs})[0]

    input_name = "data"
    shape_dict = {input_name: input_shape}
    mod2, params2 = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

    with open(f"models/{NAME}_onnx.txt", "w") as text_file:
        text_file.write(mod2.astext(show_meta_data=True))

    # Bulid the subgraph
    ctx = tvm.device("cuda", 0)

    with tvm.transform.PassContext(opt_level=3):
        lib2 = relay.build(mod2, target="cuda", target_host="llvm", params=params2)

    m2 = runtime.GraphModule(lib2["default"](ctx))
    # Set inputs
    m2.set_input(input_name, tvm.nd.array(inputs.astype(np.float32)))

    ftimer = m2.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    perf = np.mean(prof_res)
    print(NAME)
    print("%.5f ms" % (perf))

    m2.run()
    out = m2.get_output(0)
    out_tvm2 = out.asnumpy()

    result_dict[NAME] = perf
    #print(np.abs((out_onnx - out_tvm)).max())

print(result_dict)