import h5py
import lava.lib.dl.slayer as sl
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import lava.utils as utils
import torch.nn.functional as F
import torchview


class ChannelMean:
    def __call__(self, tensor):
        out = tensor[0] + tensor[1] + tensor[2]
        out = torch.div(out, 3)
        return out


class HorizontalFlip:
    def __call__(self, img):
        return TF.hflip(img)


class RGBTOGrayscaleInvert:
    def __call__(self, img):
        return TF.invert(TF.rgb_to_grayscale(img)) / 255


class Invert():
    def __call__(self, img):
        return TF.invert(img) / 255


# training_data = datasets.cifar.CIFAR10(
#     root='data/',
#     train=True,
#     transform=transforms.Compose([
#         # transforms.RandomAffine(
#         #     degrees=10,
#         #     translate=(0.05, 0.05),
#         #     scale=(0.95, 1.05),
#         #     shear=5,
#         # ),
#         transforms.ToTensor(),
#         # transforms.PILToTensor(),
#         # Invert(),
#         # ChannelMean(),
#         # RGBTOGrayscaleInvert(),
#
#     ]),
#     download=True,
# )
#

# testing_data = datasets.cifar.CIFAR10(
#     root='data/',
#     train=False,
#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.PILToTensor(),
#         # Invert(),
#         # ChannelMean(),
#         # RGBTOGrayscaleInvert(),
#     ]),
# )

# training_data = datasets.CIFAR10(
#     root='data/',
#     train=True,
#     transform=transforms.Compose([
#         # transforms.RandomAffine(
#         #     degrees=10,
#         #     translate=(0.05, 0.05),
#         #     scale=(0.95, 1.05),
#         #     shear=5,
#         # ),
#         transforms.ToTensor(),
#         # HorizontalFlip(),
#     ]),
#     download=True,
# )


testing_data = datasets.MNIST(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # HorizontalFlip(),
    ]),
)
dev = torch.device('cuda')

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


class CifarSpike3C0(Dataset):
    def __init__(self, data, timesteps):
        super(CifarSpike3C, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        img.to(dev)
        img = torch.stack([torch.bernoulli(img).to(dev) for _ in range(timesteps)], 3)
        return img, lbl

    def __len__(self):
        return len(self.data)


class CifarSpike1C(Dataset):
    def __init__(self, data, timesteps):
        super(CifarSpike1C, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        img.to(dev)
        o = torch.stack([torch.bernoulli(img).unsqueeze(0) for _ in range(timesteps)], 4)
        return o, lbl

    def __len__(self):
        return len(self.data)


class MNISTSpike(Dataset):
    def __init__(self, data, timesteps):
        super(MNISTSpike, self).__init__()
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index: int):
        img, lbl = self.data[index]
        o = torch.stack([torch.bernoulli(img) for _ in range(timesteps)], 3)
        return o, lbl

    def __len__(self):
        return len(self.data)


timesteps = 30
# trainSpike = CifarSpike3C(training_data, timesteps)
# testSpike = CifarSpike3C(testing_data, timesteps)
testSpike = MNISTSpike(testing_data, timesteps)

batch_size = 1
# train_loader = DataLoader(dataset=trainSpike, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testSpike, batch_size=batch_size, shuffle=False)

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

def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


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
            sl.block.cuba.Conv(neuron_params, 3, 32, 3, stride=2, ),  # 15x15
            sl.block.cuba.Conv(neuron_params, 32, 64, 3, stride=1, ),  # 13x13
            sl.block.cuba.Conv(neuron_params, 64, 128, 3, stride=2, ),  # 6x6
            sl.block.cuba.Conv(neuron_params, 128, 256, 3, stride=1, ),  # 4x4
            sl.block.cuba.Conv(neuron_params, 256, 64, 3, stride=2, ),  # 1x1
            sl.block.cuba.Flatten(),
            sl.block.cuba.Dense(neuron_params, 1 * 1 * 64, 128, ),
            sl.block.cuba.Dense(neuron_params, 128, 10, ),
        ])

    def forward(self, x):
        count = []
        event_cost = 0

        for block in self.blocks:
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())

        return x, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

class MNISTConvClass(torch.nn.Module):
    def __init__(self):
        super(MNISTConvClass, self).__init__()

        neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 1,
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

    # def forward(self, x):
    #     for block in self.blocks:
    #         x = block(x)
    #     return x
    #
    def forward(self, x):
        count = []
        event_cost = 0

        for block in self.blocks:
            # forward computation is as simple as calling the blocks in a loop
            x = block(x)
            if hasattr(block, 'neuron'):
                event_cost += event_rate_loss(x)
                count.append(torch.sum(torch.abs((x[..., 1:]) > 0).to(x.dtype)).item())

        return x, torch.FloatTensor(count).reshape((1, -1)).to(x.device)

    def export_hdf5(self, filename):
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


# trained_folder = "/home/ronak/Code/SpikeTrained/MNIST/22:08:49:Mar-17-23/"
# trained_folder = "/home/ronak/Code/SpikeTrained/MNIST/16:00:49:Apr-06-23/"
# trained_folder = "/home/ronak/Code/SpikeTrained/MNIST/23:55:25:Apr-06-23"
# net.load_state_dict(torch.load(trained_folder + '/network.pt'))
# net = torch.nn.DataParallel(net)

# from collections import OrderedDict
#
# new_state_dict = OrderedDict()
# for k, v in old.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# net.load_state_dict(new_state_dict)

# net.export_hdf5(trained_folder + '/network.net')
# net.to(dev)
# stats = sl.utils.LearningStats()
# stats.load(trained_folder + '/')
# stats.plot(figsize=(15, 5))
# net.eval()
# net.export_hdf5(trained_folder + '/network.net')

# with torch.no_grad():
#     corr = 0
#     for i, (inp, label) in enumerate(test_loader):
#         out = net.forward(inp.to(dev))
#         rates = torch.sum(out, 2)
#         corr += 1 if torch.argmax(rates).to(dev) == label.to(dev) else 0
#     print(f'accuracy: {corr / len(testSpike)}')

# with torch.no_grad():
#     for j in range(20, 25, 1):
#         corr = 0
#         testSpike = MNISTSpike(testing_data, timesteps=j)
#         test_loader = DataLoader(dataset=testSpike, batch_size=1, shuffle=True)
#         for i, (inp, label) in enumerate(test_loader):
#             out = net.forward(inp.to(dev))
#             rates = torch.sum(out, 2)
#             corr += 1 if torch.argmax(rates).to(dev) == label.to(dev) else 0
#         print(f'timesteps {j} : accuracy: {corr / len(testSpike)}')

# with torch.no_grad():
#     trials = 10
#     acc = 0
#     for j in range(trials):
#         corr = 0
#         testSpike = MNISTSpike(testing_data, timesteps=25)
#         test_loader = DataLoader(dataset=testSpike, batch_size=1, shuffle=True)
#         for i, (inp, label) in enumerate(test_loader):
#             out = net.forward(inp.to(dev))
#             rates = torch.sum(out, 2)
#             corr += 1 if torch.argmax(rates).to(dev) == label.to(dev) else 0
#         print(f'trial {j} : accuracy: {corr / len(testSpike)}')
#         acc += corr / len(testSpike)
#     print(f'avg. acc.:{acc / trials}')

# with torch.no_grad():
#     for i in range(1):
#         # n = np.random.randint(len(testSpike))
#         n = i
#         tensor, lbl = testing_data[n]
#         tensor = TF.hflip(tensor)
#         spike_tensor, label = testSpike[n]
#         spike_tensor = spike_tensor.squeeze(0)  # get rid of batch dimension
#         event = sl.io.tensor_to_event(spike_tensor.cpu().data.numpy())
#         anim = event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
#         anim.save(f'gifs/{labels[label]}_{i}.gif', animation.PillowWriter(fps=24), dpi=300)
#         # anim.save(f'gifs/{label}_{i}.gif', animation.PillowWriter(fps=24), dpi=300)
#         fig, axes = plt.subplots()
#         plt.imshow(np.rot90(tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1)), 3, axes=(0, 1)), cmap='RdBu')
#         axes.set_title(f'{labels[label]}_{i}')
#         # axes.set_title(f'{label}_{i}')
#         # plt.savefig(f"gifs/{label}_{i}.png")
#         plt.savefig(f"gifs/{labels[label]}_{i}.png")
#         plt.show()
#         accum = torch.abs(torch.sum(spike_tensor, -1) / spike_tensor.shape[-1])
#         plt.imshow(np.fliplr(np.rot90(accum.permute(*torch.arange(accum.ndim - 1, -1, -1)).cpu(), 3, axes=(0, 1))),
#                    cmap='RdBu')
#         # plt.savefig(f"gifs/{label}_{i}_accum.png")
#         plt.savefig(f"gifs/{labels[label]}_{i}_accum.png")
#         plt.show()

## compare ops
import torchinfo
# dev=torch.device('cpu')
# net = ConvClass()
# net.load_state_dict(
#     torch.load("/home/ronak/Code/SpikeTrained/CIFAR/15:36:46:Apr-23-23-3C/network.pt"))
# net.to(dev)

net = MNISTConvClass()
net.load_state_dict(
    torch.load("/home/ronak/Code/SpikeTrained/MNIST/16:34:06:Apr-22-23/network.pt"))
net.to(dev)
torchinfo.summary(net, input_size=(1, 1, 28, 28, 30))


# torchinfo.summary(net, input_size=(1, 3, 32, 32, 30))
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4)
stats = sl.utils.LearningStats()
error = sl.loss.SpikeRate(true_rate=0.15, false_rate=0.02, reduction='sum').to(dev)
assistant = sl.utils.Assistant(net, error, optimizer, stats, classifier=sl.classifier.Rate.predict, count_log=True)
counts = []
for i, (input, ground_truth) in enumerate(test_loader):
    # if i == 1000:
    #     break
    _, count = assistant.test(input, ground_truth)
    count = (count.flatten() / (input.shape[-1] - 1) / input.shape[0]).tolist()  # count skips first events
    counts.append(count)
    print('\rEvent count : ' + ', '.join([f'{c:.4f}' for c in count]), f'| {stats.testing}', end='')

counts = np.mean(counts, axis=0)


def compare_ops(net, counts, mse):
    shapes = [b.shape for b in net.blocks if hasattr(b, 'neuron')]

    # synops calculation
    snn_synops = []
    ann_synops = []
    for l in range(1, len(net.blocks)):
        if hasattr(net.blocks[l], 'neuron') is False:
            break
        conv_synops = (  # ignoring padding
                counts[l - 1]
                * net.blocks[l].synapse.out_channels
                * np.prod(net.blocks[l].synapse.kernel_size)
                / np.prod(net.blocks[l].synapse.stride)
        )
        snn_synops.append(conv_synops)
        ann_synops.append(conv_synops * np.prod(net.blocks[l - 1].shape) / counts[l - 1])
        # ann_synops.append(conv_synops*np.prod(net.blocks[l-1].shape)/counts[l-1]*np.prod(net.blocks[l].synapse.stride))

    for l in range(l + 1, len(net.blocks)):
        fc_synops = counts[l - 2] * net.blocks[l].synapse.out_channels
        snn_synops.append(fc_synops)
        ann_synops.append(fc_synops * np.prod(net.blocks[l - 1].shape) / counts[l - 2])

    # event and synops comparison
    total_events = np.sum(counts)
    total_synops = np.sum(snn_synops)
    total_ann_activs = np.sum([np.prod(s) for s in shapes])
    total_ann_synops = np.sum(ann_synops)
    total_neurons = np.sum([np.prod(s) for s in shapes])
    steps_per_inference = 1

    print(f'|{"-" * 77}|')
    print('|', ' ' * 23, '|          SNN           |           ANN           |')
    print(f'|{"-" * 77}|')
    print('|', ' ' * 7, f'|     Shape     |  Events  |    Synops    | Activations|    MACs    |')
    print(f'|{"-" * 77}|')
    for l in range(len(counts)):
        print(f'| layer-{l} | ', end='')
        if len(shapes[l]) == 3:
            z, y, x = shapes[l]
        elif len(shapes[l]) == 1:
            z = shapes[l][0]
            y = x = 1
        print(f'({x:-3d},{y:-3d},{z:-3d}) | {counts[l]:8.2f} | ', end='')
        if l == 0:
            print(f'{" " * 12} | {np.prod(shapes[l]):-10.0f} | {" " * 10} |')
        else:
            print(f'{snn_synops[l - 1]:12.2f} | {np.prod(shapes[l]):10.0f} | {ann_synops[l - 1]:10.0f} |')
    print(f'|{"-" * 77}|')
    print(
        f'|  Total  | {" " * 13} | {total_events:8.2f} | {total_synops:12.2f} | {total_ann_activs:10.0f} | {total_ann_synops:10.0f} |')
    print(f'|{"-" * 77}|')

    print('\n')
    print(f'MSE            : {mse:.5} sq. radians')
    print(f'Total neurons  : {total_neurons}')
    print(f'Events sparsity: {total_ann_activs / total_events:5.2f}x')
    print(f'Synops sparsity: {total_ann_synops / total_synops:5.2f}x')


mse = 0.000536
#cifar_mse = 0.042948
compare_ops(net, counts, mse=mse)

train_acc = [
    0.212120,
    0.307860,
    0.357120,
    0.417940,
    0.427280,
    0.430680,
    0.434780,
    0.439760,
    0.445460,
    0.449860,
]

epochs = range(len(train_acc))

test_acc = [
    0.279600,
    0.341300,
    0.362600,
    0.459400,
    0.435300,
    0.444400,
    0.455000,
    0.465900,
    0.480000,
    0.466500
]
train_loss = [
    0.063238,
    0.052027,
    0.049243,
    0.045049,
    0.045024,
    0.045224,
    0.045513,
    0.044887,
    0.044197,
    0.043924,
]
test_loss = [
    0.053953,
    0.054274,
    0.048102,
    0.043782,
    0.044988,
    0.046421,
    0.044192,
    0.046293,
    0.043072,
    0.042948,
]

# plt.plot(epochs, train_acc, test_acc, label="Train and Test Accuracies")
# plt.title("Train and Test Accuracies")
# plt.xlabel("Epochs")
# plt.legend(["Train", "Test"])
# plt.savefig(f"./CIFAR_ACC.png")
# plt.show()
# plt.plot(epochs, test_acc, test_loss, label="Train and Test Loss")
# plt.title("Train and Test Loss")
# plt.legend(["Train", "Test"])
# plt.xlabel("Epochs")
# plt.savefig(f"./CIFAR_LOSS.png")
# plt.show()