import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from lava.lib.dl import netx, slayer
import lava.lib.dl.slayer as sl
from lava.proc import io
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
import h5py
import numpy as np
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MNISTClass(torch.nn.Module):
    def __init__(self):
        super(MNISTClass, self).__init__()

        neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }

        self.blocks = torch.nn.ModuleList([
            sl.block.cuba.Conv(neuron_params, 1, 32, 3, stride=2, ),  # 13x13
            sl.block.cuba.Conv(neuron_params, 32, 64, 3, stride=1, ),  # 11x11
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=2, ),  # 5x5
            sl.block.cuba.Conv(neuron_params, 128, 128, 3, stride=1, ),  # 3x3
            sl.block.cuba.Conv(neuron_params, 128, 64, 3, stride=2, ),  # 1x1
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params, 1 * 1 * 64, 64, ),
            sl.block.cuba.Dense(neuron_params, 64, 10, ),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


class MNISTSpike(Dataset):
    def __init__(self, data, timesteps):
        super(MNISTSpike, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        o = torch.stack([torch.bernoulli(img) for _ in range(self.timesteps)], 3)
        return o, lbl

    def __len__(self):
        return len(self.data)


timesteps = 30

testing_data = datasets.MNIST(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
    download=True
)
testSpike = MNISTSpike(testing_data, timesteps)
test_loader = DataLoader(dataset=testSpike, batch_size=len(testing_data), shuffle=True)

dataset = next(iter(test_loader))[0].numpy()
targets = next(iter(test_loader))[1].numpy()
lbl = []
out = []


def main():
    net_ladl = MNISTClass()

    net_ladl.load_state_dict(torch.load("/home/ronak/Code/SpikeTrained/MNIST/16:34:06:Apr-22-23/network.pt"))

    net_lava = netx.hdf5.Network(net_config='/home/ronak/Code/SpikeTrained/MNIST/16:34:06:Apr-22-23/network.net',
                                 input_shape=(28, 28, 1))

    for i, img in enumerate(dataset):
        x = torch.from_numpy(img)
        label = targets[i]
        print(f'label = {label}')
        lbl.append(label)
        with torch.no_grad():
            y = net_ladl(x.unsqueeze(0))
            print(y.sum(-1))
            print('y(ladl)=', y.sum(-1).argmax().item())

        x = x.permute(2, 1, 0, 3)
        x = x.numpy()
        source = io.source.RingBuffer(data=x)
        sink = io.sink.RingBuffer(shape=(10,), buffer=timesteps)
        source.s_out.connect(net_lava.inp)
        net_lava.out.connect(sink.a_in)
        run_condition = RunSteps(num_steps=timesteps)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')
        net_lava.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        net_lava.stop()
        print(output.sum(-1))
        print('y(lava)=', output.sum(-1).argmax())
        out.append(output.sum(-1).argmax())
        print("")


if __name__ == '__main__':
    main()
