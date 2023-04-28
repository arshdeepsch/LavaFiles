import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os
import cv2
from torchinfo import summary
import torchvision.transforms.functional as TF
import torchvision as tv
from PIL import Image
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as nnF
import torchaudio.transforms as T
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from lava.utils.profiler import Profiler
import random
import shutil
import h5py
import lava.lib.dl.slayer as sl
import lava.proc.io.dataloader as dloader
import lava.lib.dl.netx as netx
from lava.magma.core.run_conditions import RunSteps
import lava.proc.io as io
from lava.lib.dl import slayer
import lava.lib.dl.slayer.utils.time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg, Loihi1SimCfg, Loihi1HwCfg

testing_data = datasets.MNIST(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

class CustomRunConfig(Loihi2SimCfg):
    def select(self, proc, proc_models):
        # customize run config to always use float model for io.sink.RingBuffer
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        else:
            return super().select(proc, proc_models)

class MNISTSpike(Dataset):
    def __init__(self, data, timesteps):
        super(MNISTSpike, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        o = torch.stack([torch.bernoulli(img) for _ in range(self.timesteps)], 3)
        o = o.unsqueeze(-2)
        o = o.squeeze(0)
        return o, lbl

    def __len__(self):
        return len(self.data)


timesteps = 30
testSpikes = MNISTSpike(data=testing_data, timesteps=timesteps)
data = []
for i in range(10):
    data.append(testSpikes[i])

# old = "/home/ronak/Code/SpikeTrained/MNIST/22:08:49:Mar-17-23/network.net"
old = "/home/ronak/Code/SpikeTrained/MNIST/20:39:23:Apr-12-23/network.net"
# old = "/home/ronak/Code/SpikeTrained/MNIST/16:00:49:Apr-06-23/network.net"
network = netx.hdf5.Network(net_config=old, input_shape=(28, 28, 1))
network.reset_interval = len(network)+1
print(network)
num_samples = 10
# steps_per_sample = network.reset_interval
# num_steps = num_samples * steps_per_sample + 1
# readout_offset = (steps_per_sample - 1) + len(network.layers)

# steps_per_sample = network.reset_interval
# readout_offset = (steps_per_sample-1) + len(network.layers)
# num_steps = num_samples*steps_per_sample

steps_per_sample = timesteps
readout_offset = steps_per_sample - 1 + len(network.layers)
num_steps = num_samples * steps_per_sample
for i, l in enumerate(network.layers):
    u_resetter = io.reset.Reset(interval=steps_per_sample, offset=i)
    v_resetter = io.reset.Reset(interval=steps_per_sample, offset=i)
    u_resetter.connect_var(l.neuron.u)
    v_resetter.connect_var(l.neuron.v)

run_config = CustomRunConfig(select_tag="fixed_pt")
input_loader = io.dataloader.SpikeDataloader(dataset=data, interval=steps_per_sample)
output = lava.proc.io.sink.RingBuffer(shape=network.out_layer.shape, buffer=num_steps)
gt_logger = io.sink.RingBuffer(shape=(1,), buffer=num_steps)
input_loader.s_out.connect(network.inp)
network.out.connect(output.a_in)
input_loader.ground_truth.connect(gt_logger.a_in)
network.run(condition=RunSteps(num_steps=num_steps), run_cfg=run_config)
out = output.data.get()
out = torch.Tensor(out)
gts = gt_logger.data.get().flatten()[::steps_per_sample]
network.stop()
print(f"ground_truth: {gts}")
print(f"out: {out} \n shape:{out.shape}")
rates = torch.sum(out, 1)
rates_out = []
for i in range(1, 10):
    result = out[:, i * readout_offset:((i + 1) * readout_offset)]
    rates_out.append(torch.sum(result, 1))

print("Cumulative Rates")
print(rates)
print("Rates for each sample")

for rate in rates_out:
    print(rate)

output_arr = out.reshape(10, num_samples, timesteps)
classifier = slayer.classifier.Rate()
prediction = classifier.predict(output_arr)
print(gts)
print('Accuracy', np.sum(gts == prediction) / len(prediction))
