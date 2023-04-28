import h5py
import lava.lib.dl.slayer as sl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os


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

        neuron_params_drop = {**neuron_params, 'dropout': sl.neuron.Dropout(p=0.05)}

        self.blocks = torch.nn.ModuleList([
            sl.block.cuba.Input(neuron_params, weight=1, bias=0),
            sl.block.cuba.Conv(neuron_params, 1, 32, 3, stride=2, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 32, 64, 3, stride=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=2, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 128, 256, 1, stride=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 256, 512, 1, stride=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 512, 64, 1, stride=1, weight_norm=True),
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params_drop, 6 * 6 * 64, 100, weight_norm=True),
            sl.block.cuba.Dense(neuron_params_drop, 100, 10, weight_norm=True),
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


class ConvClass0(torch.nn.Module):
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

        neuron_params_drop = {**neuron_params, 'dropout': sl.neuron.Dropout(p=0.05)}

        self.blocks = torch.nn.ModuleList([
            sl.block.cuba.Input(neuron_params, weight=1, bias=0),

            sl.block.cuba.Conv(neuron_params, 1, 32, 3, padding=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 32, 32, 3, stride=2, padding=1, weight_norm=True),

            sl.block.cuba.Conv(neuron_params, 32, 64, 3, padding=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 64, 64, 3, stride=2, padding=1, weight_norm=True),

            sl.block.cuba.Conv(neuron_params, 64, 128, 3, padding=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 128, 128, 3, stride=2, padding=1, weight_norm=True),

            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params_drop, 4 * 4 * 128, 512, weight_norm=True),
            sl.block.cuba.Dense(neuron_params_drop, 512, 10, weight_norm=True),
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


class ChannelMean(object):
    def __call__(self, tensor):
        out = tensor[0] + tensor[1] + tensor[2]
        out = out / 3
        return out.unsqueeze(0)


class RGBTOGrayscale:
    def __call__(self, img):
        return TF.rgb_to_grayscale(img) / 255


training_data = datasets.cifar.CIFAR10(
    root='data/',
    train=True,
    transform=transforms.Compose([
        # transforms.RandomAffine(
        #     degrees=10,
        #     translate=(0.05, 0.05),
        #     scale=(0.95, 1.05),
        #     shear=5,
        # ),
        transforms.ToTensor(),
        # RGBTOGrayscale(),
        ChannelMean(),
    ]),
    download=True,
)

testing_data = datasets.cifar.CIFAR10(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # RGBTOGrayscale(),
        ChannelMean(),
    ]),
)


class CifarSpike1C(Dataset):
    def __init__(self, data, timesteps):
        super(CifarSpike1C, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        o = torch.stack([torch.bernoulli(img) for _ in range(timesteps)], 3)
        return o, lbl

    def __len__(self):
        return len(self.data)


dev = torch.device('cuda')
net = ConvClass()
# net = torch.nn.DataParallel(net)
net.to(dev)
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)

timesteps = 50
trainSpike = CifarSpike1C(training_data, timesteps)
testSpike = CifarSpike1C(testing_data, timesteps)

stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(dev)
batch_size = 1
train_loader = DataLoader(dataset=trainSpike, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSpike, batch_size=batch_size, shuffle=False)
assistant = sl.utils.Assistant(net, error, optimizer, stats, classifier=sl.classifier.Rate.predict)

labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

now = datetime.now()
currT = now.strftime("%X:%b-%d-%y")
epochs = 10
trained_folder = f'/home/ronak/Code/SpikeTrained/{currT}-1C'
try:
    with torch.cuda.device(dev):
        os.makedirs(trained_folder)
        print(f'{currT}\n')
        for epoch in range(epochs):
            for i, (inp, label) in enumerate(train_loader):
                net.train()
                output = assistant.train(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            for i, (inp, label) in enumerate(test_loader):
                net.eval()
                output = assistant.test(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            if epoch % 5 == 4:  # cleanup display
                print('\r', ' ' * len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
                stats_str = str(stats).replace("| ", "\n")
                print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

            if stats.testing.best_loss:
                torch.save(net.state_dict(), trained_folder + '/network.pt')
                net.export_hdf5(trained_folder + '/network.net')
            stats.update()
            stats.save(trained_folder + '/')
            # net.grad_flow(trained_folder + '/')
finally:
    os.rmdir(trained_folder)
    print("\nRemoved Folder")
