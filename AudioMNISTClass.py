import shutil
import h5py
import lava.lib.dl.slayer as sl
import lava.lib.dl.slayer.utils.time
import pandas as pd
import torch
import torch
import torchvision as tv
from PIL import Image
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as nnF
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os
import cv2
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
        neuron_params_graded = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
            # 'graded_spike': True,
        }

        neuron_params_drop = {**neuron_params, 'dropout': sl.neuron.Dropout(p=0.05)}

        self.blocks = torch.nn.ModuleList([
            # sl.block.cuba.Input(neuron_params, weight=1, bias=0),
            sl.block.cuba.Conv(neuron_params, 1, 64, 3, stride=2, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 128, 256, 3, stride=2, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 256, 256, 1, stride=1, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 256, 64, 1, stride=2, weight_norm=True),
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params_drop, 7 * 7 * 64, 100, weight_norm=True),
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
        neuron_params_graded = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
            # 'graded_spike': True,
        }

        neuron_params_drop = {**neuron_params, 'dropout': sl.neuron.Dropout(p=0.05)}

        self.blocks = torch.nn.ModuleList([
            # sl.block.cuba.Input(neuron_params, weight=1, bias=0),
            sl.block.cuba.Conv(neuron_params, 1, 64, 3, stride=2, weight_norm=True),
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=1, weight_norm=True, groups=64),
            sl.block.cuba.Conv(neuron_params, 128, 256, 3, stride=2, weight_norm=True, groups=128),
            sl.block.cuba.Conv(neuron_params, 256, 256, 1, stride=1, weight_norm=True, groups=256),
            sl.block.cuba.Conv(neuron_params, 256, 512, 1, stride=2, weight_norm=True, groups=256),
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params_drop, 7 * 7 * 512, 100, weight_norm=True),
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


dev = torch.device('cuda')
net = ConvClass()
summary(net, (1, 1, 64, 64, 25))
net.to(dev)
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-5)

TRAIN_DIR = "/home/ronak/Code/data/free-spoken-digit-dataset-master/training-spectrograms/"
TEST_DIR = "/home/ronak/Code/data/free-spoken-digit-dataset-master/testing-spectrograms/"


class SpecTrain(Dataset):
    def __init__(self, data_dir, timesteps):
        super().__init__()
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.len = len(self.files)
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        file = self.files[index]
        lbl = int(file[0])
        image = Image.open(self.data_dir + file, formats=["png"])
        image = transforms.Grayscale()(image)
        image = transforms.ToTensor()(image)
        image = lava.lib.dl.slayer.utils.time.replicate(image, self.timesteps)
        return image, lbl

    def __len__(self):
        return self.len


class SpecTest(Dataset):
    def __init__(self, data_dir, timesteps):
        super().__init__()
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.len = len(self.files)
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        file = self.files[index]
        lbl = int(file[0])
        image = Image.open(self.data_dir + file, formats=["png"])
        image = transforms.Grayscale()(image)
        image = transforms.ToTensor()(image)
        image = lava.lib.dl.slayer.utils.time.replicate(image, self.timesteps)
        return image, lbl

    def __len__(self):
        return self.len


timesteps = 25
trainD = SpecTrain(TRAIN_DIR, timesteps)
testD = SpecTest(TEST_DIR, timesteps)

stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.2, false_rate=0.02, reduction='sum').to(dev)
batch_size = 1
train_loader = DataLoader(dataset=trainD, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testD, batch_size=batch_size, shuffle=True)
assistant = sl.utils.Assistant(net, error, optimizer, stats, classifier=sl.classifier.Rate.predict)
now = datetime.now()
currT = now.strftime("%X:%b-%d-%y")
epochs = 20
trained_folder = f'/home/ronak/Code/SpikeTrained/AudioMNISTClass/{currT}/'
classifier = sl.classifier.Rate.predict

#
# def error(inp, categ_label, true_rate=0.2, false_rate=0.03, reduction='sum'):
#     spike_rate = sl.classifier.Rate.rate(inp)
#     target_rate = true_rate * categ_label \
#                   + false_rate * (1 - categ_label)
#     return nnF.mse_loss(
#         spike_rate.flatten(),
#         target_rate.flatten(),
#         reduction=reduction
#     )
#
#
# def train_step(inp, label):
#     net.train()
#     inp = inp.to(dev)
#     label = label.to(dev)
#     output = net(inp)
#     loss = error(output, label)
#     if stats is not None:
#         stats.training.num_samples += inp.shape[0]
#         stats.training.loss_sum += loss.cpu().data.item() \
#                                    * output.shape[0]
#         if classifier is not None:  # classification
#             stats.training.correct_samples += torch.sum(
#                 classifier(output) == label
#             ).cpu().data.item()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return output
#
#
# def test_step(inp, label):
#     net.eval()
#     with torch.no_grad():
#         inp = inp.to(dev)
#         label = label.to(dev)
#         output = net(inp)
#         loss = error(output, label)
#         if stats is not None:
#             stats.testing.num_samples += inp.shape[0]
#             stats.testing.loss_sum += loss.cpu().data.item() \
#                                       * output.shape[0]
#             if classifier is not None:  # classification
#                 stats.testing.correct_samples += torch.sum(
#                     classifier(output) == label
#                 ).cpu().data.item()
#     return output


with torch.cuda.device(dev):
    os.makedirs(trained_folder)
    print(f'{currT}\n')
    try:
        for epoch in range(epochs):
            if epoch == 5:
                assistant.reduce_lr(1e2)

            for i, (inp, label) in enumerate(train_loader):
                output = assistant.train(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            for i, (inp, label) in enumerate(test_loader):
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

    except Exception as inst:
        print(inst)
        shutil.rmtree(trained_folder)
        os.rmdir(trained_folder)
        print("\nRemoved Folder")
