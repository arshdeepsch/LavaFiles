import shutil

import h5py
import lava.lib.dl.slayer as sl
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os
from torchinfo import summary


class ConvClass(torch.nn.Module):
    def __init__(self):
        super(ConvClass, self).__init__()

        neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }

        self.blocks = torch.nn.ModuleList([
            sl.block.cuba.Conv(neuron_params, 1, 32, 3, stride=2,),  # 13x13
            sl.block.cuba.Conv(neuron_params, 32, 64, 3, stride=1,),  # 11x11
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=2,),  # 5x5
            sl.block.cuba.Conv(neuron_params, 128, 128, 3, stride=1,),  # 3x3
            sl.block.cuba.Conv(neuron_params, 128, 64, 3, stride=2,),  # 1x1
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params, 1 * 1 * 64, 64,),
            sl.block.cuba.Dense(neuron_params, 64, 10,),
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


training_data = datasets.MNIST(
    root='data/',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
    download=True,
)

testing_data = datasets.MNIST(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)


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


timesteps = 25
batch_size = 4
dev = torch.device('cuda')
net = ConvClass().to(dev)
summary(net, (batch_size, 1, 28, 28, timesteps))
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)

trainSpike = MNISTSpike(training_data, timesteps)
testSpike = MNISTSpike(testing_data, timesteps)

stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='mean').to(dev)
train_loader = DataLoader(dataset=trainSpike, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSpike, batch_size=batch_size, shuffle=True)
assistant = sl.utils.Assistant(net, error, optimizer, stats, classifier=sl.classifier.Rate.predict)

now = datetime.now()
currT = now.strftime("%X:%b-%d-%y")
epochs = 5
trained_folder = f'/home/ronak/Code/SpikeTrained/MNIST/{currT}'
os.makedirs(trained_folder)
checkpoints = [2, 4]
try:
    with torch.cuda.device(dev):
        print(f'{currT}\n')
        for epoch in range(epochs):
            if epoch in checkpoints:
                assistant.reduce_lr(1/10)

            for i, (inp, label) in enumerate(train_loader):
                inp.to(dev)
                output = assistant.train(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            for i, (inp, label) in enumerate(test_loader):
                inp.to(dev)
                output = assistant.test(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            if epoch % epochs / 2 == epochs / 2 - 1:  # cleanup display
                print('\r', ' ' * len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
                stats_str = str(stats).replace("| ", "\n")
                print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

            if stats.testing.best_accuracy:
                torch.save(net.state_dict(), trained_folder + f'/network_{stats.training.max_accuracy}.pt')
                net.export_hdf5(trained_folder + f'/network_{stats.training.max_accuracy}.net')
                print(f"\nModel saved")

            stats.update()
            stats.save(trained_folder + '/')
except:
    if not os.path.exists(f"/home/ronak/Code/SpikeTrained/MNIST/{currT}/network_{stats.training.max_accuracy}.pt"):
        shutil.rmtree(trained_folder)
        # os.rmdir(trained_folder)
        print("\nRemoved Folder")
