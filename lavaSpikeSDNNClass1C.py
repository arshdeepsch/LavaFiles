import h5py
import lava.lib.dl.slayer as sl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import os


class ConvClass(torch.nn.Module):
    def __init__(self):
        super(ConvClass, self).__init__()
        neuron_params = {
            'threshold': 0.1,
            'tau_grad': 0.5,
            'scale_grad': 1,
            'requires_grad': True,
            'shared_param': True,
            'activation': F.relu,
        }

        neuron_params_drop = {**neuron_params, 'dropout': sl.neuron.Dropout(p=0.05)}
        self.blocks = torch.nn.ModuleList([
            sl.block.sigma_delta.Input(neuron_params, weight=1, bias=0),
            sl.block.sigma_delta.Conv(neuron_params, 1, 32, 3, stride=2, weight_norm=True),
            sl.block.sigma_delta.Conv(neuron_params, 32, 64, 3, stride=1, weight_norm=True),
            sl.block.sigma_delta.Conv(neuron_params, 64, 128, 3, stride=2, weight_norm=True),
            sl.block.sigma_delta.Conv(neuron_params, 128, 256, 1, stride=1, weight_norm=True),
            sl.block.sigma_delta.Conv(neuron_params, 256, 512, 1, stride=1, weight_norm=True),
            sl.block.sigma_delta.Conv(neuron_params, 512, 64, 1, stride=1, weight_norm=True),
            sl.block.sigma_delta.Flatten(),
            sl.block.sigma_delta.Dense(neuron_params_drop, 6 * 6 * 64, 100, weight_norm=True),
            sl.block.sigma_delta.Dense(neuron_params_drop, 100, 10, weight_norm=True),
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
# net = torch.nn.DataParallel(net)
# net.to(dev)
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)


class ChannelMean(object):
    def __call__(self, tensor):
        out = tensor[0] + tensor[1] + tensor[2]
        out = out / 3
        return out.unsqueeze(0)


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
        ChannelMean(),
    ]),
    download=True,
)

testing_data = datasets.cifar.CIFAR10(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
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


timesteps = 1000
trainSpike = CifarSpike1C(training_data, timesteps)
testSpike = CifarSpike1C(testing_data, timesteps)

stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.2, false_rate=0.03).to(dev)
batch_size = 1
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


class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# eS = EarlyStopping(patience=5, min_delta=10)
now = datetime.now()
currT = now.strftime("%X:%b-%d-%y")
epochs = 10
trained_folder = f'/home/ronak/Code/SpikeTrainedSDNN/{currT}-1C'
try:
    with torch.cuda.device(dev):
        os.makedirs(trained_folder)
        print(f'{currT}\n')
        for epoch in range(epochs):
            for i, (inp, label) in enumerate(train_loader):
                inp.to(dev)
                label.to(dev)
                net.train()
                output = assistant.train(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            for i, (inp, label) in enumerate(test_loader):
                inp.to(dev)
                label.to(dev)
                net.eval()
                output = assistant.test(inp, label)
                print(f'\r[Epoch {epoch:3d}/{epochs}] {stats}', end='')

            if epoch % 20 == 19:  # cleanup display
                print('\r', ' ' * len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
                stats_str = str(stats).replace("| ", "\n")
                print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')
                stats.validation.loss()
            if stats.testing.best_loss:
                torch.save(net.state_dict(), trained_folder + '/network.pt')
            stats.update()
            stats.save(trained_folder + '/')
            # net.grad_flow(trained_folder + '/')
finally:
    os.rmdir(trained_folder)
    print("\nRemoved Folder")
