import shutil
import h5py
import lava.lib.dl.slayer as sl
import torch
# import torchaudio
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os
import cv2
import torchinfo

class ConvClass(torch.nn.Module):
    def __init__(self):
        super(ConvClass, self).__init__()

        neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 1,
            'requires_grad': True,
        }

        self.blocks = torch.nn.ModuleList([
            sl.block.cuba.Conv(neuron_params, 3, 64, 3, stride=2, ),  # 15
            sl.block.cuba.Conv(neuron_params, 64, 72, 3, stride=1, ),  # 13
            sl.block.cuba.Conv(neuron_params, 72, 256, 3, stride=2, ),  # 6
            sl.block.cuba.Conv(neuron_params, 256, 256, 1, stride=1, ),  # 6
            sl.block.cuba.Conv(neuron_params, 256, 64, 1, stride=1, ),  # 6
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params, 6 * 6 * 64, 100),
            sl.block.cuba.Dense(neuron_params, 100, 10),
        ])

        # self.blocks = torch.nn.ModuleList([
        #     sl.block.cuba.Conv(neuron_params, 3, 32, 3, stride=2, ),  # 15x15
        #     sl.block.cuba.Conv(neuron_params, 32, 64, 3, stride=1, ),  # 13x13
        #     sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=2, ),  # 6x6
        #     sl.block.cuba.Conv(neuron_params, 128, 256, 3, stride=1, ),  # 4x4
        #     sl.block.cuba.Conv(neuron_params, 256, 64, 3, stride=2, ),  # 1x1
        #     sl.block.cuba.Flatten(),
        #     sl.block.cuba.Dense(neuron_params, 1 * 1 * 64, 128, ),
        #     sl.block.cuba.Dense(neuron_params, 128, 10, ),
        # ])

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
        out = torch.div(out, 3)
        return out


training_data = datasets.cifar.CIFAR10(
    root='data/',
    train=True,
    transform=transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5,
        ),
        transforms.RandomRotation(25),
        # transforms.RandAugment(num_ops=2),
        transforms.ToTensor(),
        # transforms.PILToTensor(),
        # Invert(),
        # ChannelMean(),
        # RGBTOGrayscaleInvert(),

    ]),
    download=True,
)

testing_data = datasets.cifar.CIFAR10(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.PILToTensor(),
        # Invert(),
        # ChannelMean(),
        # RGBTOGrayscaleInvert(),
    ]),
)


class CifarSpike3C(Dataset):
    def __init__(self, data, timesteps):
        super(CifarSpike3C, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        img = img.float()
        img[0, :, :] = img[0, :, :] - 0.4913997551666284
        img[1, :, :] = img[1, :, :] - 0.48215855929893703
        img[2, :, :] = img[2, :, :] - 0.4465309133731618
        # img[0, :, :]= img[0, :, :] / 0.24703225141799082
        # img[1, :, :] = img[1, :, :] / 0.24348516474564
        # img[2, :, :] = img[2, :, :] / 0.26158783926049628
        sign = torch.sign(img)
        torch.abs_(img)
        img = torch.stack([torch.bernoulli(img).to(dev) * sign.to(dev) for _ in range(timesteps)], 3)
        return img, lbl

    def __len__(self):
        return len(self.data)


# class CifarSpike3C(Dataset):
#     def __init__(self, data, timesteps):
#         super(CifarSpike3C, self).__init__()
#         self.data = data
#         self.timesteps = timesteps
#
#     def __getitem__(self, index: int):
#         img, lbl = self.data[index]
#         img.to(dev)
#         # img = torch.stack(
#         #     [torch.bernoulli(img).to(dev) for _ in range(timesteps)], 3)
#         img = torch.stack(
#             [img for _ in range(timesteps)], 3)
#         return img, lbl
#
#     def __len__(self):
#         return len(self.data)

# class Cifar3C(Dataset):
#     def __init__(self, data, timesteps):
#         super(Cifar3C, self).__init__()
#         self.data = data
#         self.timesteps = timesteps
#
#     def __getitem__(self, index: int):
#         img, lbl = self.data[index]
#         img.to(dev)
#         img = sl.utils.time.replicate(img, self.timesteps)
#         # img = torch.stack(
#         #     [torch.bernoulli(img).to(dev) for _ in range(timesteps)], 3)
#         # img = torch.stack(
#         #     [img for _ in range(timesteps)], 3)
#         return img, lbl
#
#     def __len__(self):
#         return len(self.data)


timesteps = 30
trainSpike = CifarSpike3C(training_data, timesteps)
testSpike = CifarSpike3C(testing_data, timesteps)
# trainD = Cifar3C(training_data, timesteps)
# testD = Cifar3C(testing_data, timesteps)
dev = torch.device('cuda')
net = ConvClass()
net.to(dev)
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)

stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.15, false_rate=0.02, reduction='sum').to(dev)
batch_size = 1
torchinfo.summary(net, input_size=(batch_size, 3, 32, 32, timesteps))
# train_loader = DataLoader(dataset=trainSpike, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=testSpike, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=trainSpike, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSpike, batch_size=batch_size, shuffle=True)

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
trained_folder = f'/home/ronak/Code/SpikeTrained/CIFAR/{currT}-3C/'
checkpoints = [3, 6]

try:
    with torch.cuda.device(dev):
        os.makedirs(trained_folder)
        print(f'{currT}\n')
        for epoch in range(epochs):
            if epoch in checkpoints:
                assistant.reduce_lr(10)

            for i, (inp, label) in enumerate(train_loader):
                inp.to(dev)
                output = assistant.train(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            for i, (inp, label) in enumerate(test_loader):
                inp.to(dev)
                output = assistant.valid(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            if epoch % epochs / 2 == epochs / 2 - 1:  # cleanup display
                print('\r', ' ' * len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
                stats_str = str(stats).replace("| ", "\n")
                print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

            if stats.testing.best_accuracy:
                torch.save(net.state_dict(), trained_folder + '/network.pt')
                net.export_hdf5(trained_folder + '/network.net')
                print(f"\nModel saved")

            stats.update()
            stats.save(trained_folder + '/')
except:
    if not os.path.exists(f"{trained_folder}/network.pt"):
        shutil.rmtree(trained_folder)
        # os.rmdir(trained_folder)
        print("\nRemoved Folder")
