import torch.nn as nn
import torch
import math
import argparse
from tqdm import tqdm

# This enables the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware, e.g., wingrad conv op
torch.backends.cudnn.benchmark = True
NAME = 'dcgan'
batch_size = 1
latent_dim = 100
img_size = 256
channels = 3

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# opt = parser.parse_args()
# print(opt)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.ReLU(inplace=True)]#nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            # if bn:
            #     block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = Generator()#.cuda()
        self.discriminator = Discriminator()#.cuda()

    def forward(self, z):
        # Generate a batch of images
        gen_imgs = self.generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_out = self.discriminator(gen_imgs)

        return g_out



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=5000)
    parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=1000)
    args = parser.parse_args()

    model = DCGAN().cuda()
    model.eval()
    inputs = torch.randn(1, latent_dim).cuda()

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

    # input_shape = [1, latent_dim]
    # input_data = torch.randn(input_shape)
    # scripted_model = torch.jit.trace(model.cpu(), input_data).eval()
    #
    # torch.jit.save(scripted_model, f'models/{NAME}.pth')
    #
    # input_name = "input0"
    # shape_list = [(input_name, input_shape)]
    # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    #
    # # print("Relay module function:\n", mod.astext(show_meta_data=True))
    #
    # with open(f"models/{NAME}.txt", "w") as text_file:
    #     text_file.write(mod.astext(show_meta_data=True))
    #
    # input_names = ["input0"]
    # output_names = ["output0"]
    #
    # model.eval()
    #
    # with torch.no_grad():
    #     out_torch = model(inputs.cpu()).cpu().detach().numpy()
    #
    # torch.onnx.export(scripted_model, input_data,
    #                   f"models/{NAME}.onnx", verbose=False,
    #                   export_params=True,
    #                   do_constant_folding=False,
    #                   input_names=input_names, output_names=output_names,
    #                   training=torch.onnx.TrainingMode.TRAINING,
    #                   # example_outputs=torch.rand((1, 1280, 4, 4)),
    #                   opset_version=12)
    # onnx_model = onnx.load(f"models/{NAME}.onnx")
    #
    # sess = onnxruntime.InferenceSession(f"models/{NAME}.onnx")
    # out_onnx = sess.run(["output0"], {"input0": inputs.cpu().numpy()})[0]
    #
    # input_name = "input0"
    # shape_dict = {input_name: input_shape}
    # mod2, params2 = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    #
    # with open(f"models/{NAME}_onnx.txt", "w") as text_file:
    #     text_file.write(mod2.astext(show_meta_data=True))
    #
    # # Bulid the subgraph
    # ctx = tvm.device("cuda", 0)
    #
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target="cuda", target_host="llvm", params=params)
    #
    # with tvm.transform.PassContext(opt_level=3):
    #     lib2 = relay.build(mod2, target="cuda", target_host="llvm", params=params2)
    #
    # m = runtime.GraphModule(lib["default"](ctx))
    # # Set inputs
    # m.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))
    #
    # m2 = runtime.GraphModule(lib2["default"](ctx))
    # # Set inputs
    # m2.set_input(input_name, tvm.nd.array(inputs.cpu().numpy().astype(np.float32)))
    #
    # # Measure performance
    # ftimer = m.module.time_evaluator("run", ctx, number=100, repeat=3)
    # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    # perf = np.mean(prof_res)
    # print("%.5f ms" % (perf))
    #
    # ftimer = m2.module.time_evaluator("run", ctx, number=100, repeat=3)
    # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    # perf = np.mean(prof_res)
    # print("%.5f ms" % (perf))
    #
    # m.run()
    # out = m.get_output(0)
    # out_tvm = out.asnumpy()
    #
    # m2.run()
    # out = m2.get_output(0)
    # out_tvm2 = out.asnumpy()
    #
    # print(out_tvm[0, :10, 0, 0])
    # print(out_tvm2[0, :10, 0, 0])
    # print(out_torch[0, :10, 0, 0])
    # print(out_onnx[0, :10, 0, 0])
    # TOL = 1e-01
    # assert np.allclose(out_onnx, out_torch, rtol=TOL, atol=TOL)
    # assert np.allclose(out_onnx, out_tvm, rtol=TOL, atol=TOL)
    # assert np.allclose(out_torch, out_tvm, rtol=TOL, atol=TOL)
    # assert np.allclose(out_onnx, out_tvm2, rtol=TOL, atol=TOL)
    # assert np.allclose(out_torch, out_tvm2, rtol=TOL, atol=TOL)
    #
    # print(np.abs((out_torch - out_tvm)).max())
