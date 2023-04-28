import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

training_data = datasets.CIFAR10(
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
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
        # HorizontalFlip(),
    ]),
    download=True
)

testing_data = datasets.cifar.CIFAR10(
    root='data/',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
        # transforms.PILToTensor(),
        # Invert(),
        # ChannelMean(),
        # RGBTOGrayscaleInvert(),
    ]),
)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.blocks = torch.nn.ModuleList([
            nn.Conv2d(3, 32, 3, stride=2),  # 13x13
            nn.Conv2d(32, 64, 3, stride=1),  # 11x11
            nn.Conv2d(64, 128, 3, stride=2),  # 5x5
            nn.Conv2d(128, 256, 3, stride=1),  # 3x3
            nn.Conv2d(256, 64, 3, stride=2),  # 1x1
            nn.Flatten(),
            nn.Linear(1 * 1 * 64, 128, ),
            nn.Linear(128, 10, ),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


bs = 128
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=bs, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=bs, shuffle=True)
device = torch.device('cuda')


def train(model, dataloader, bs):
    model.train()
    train_loss, train_correct = 0.0, 0.0
    num_batches = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = error(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_correct += (out.argmax(1) == y).type(torch.float).sum().item()
    return train_loss, train_correct / (bs * num_batches)


def validate(model, dataloader, bs):
    model.eval()
    test_loss, test_correct = 0.0, 0.0
    num_batches = len(dataloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = error(out, y)
            test_loss += loss.item()
            test_correct += (out.argmax(1) == y).type(torch.float).sum().item()
    return test_loss, test_correct / (bs * num_batches)


model = Model()
model.to(device)
num_epochs = 50
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
error = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 25], gamma=0.1)

for epochs in range(num_epochs):
    train_loss, train_correct = train(model, train_dataloader, bs)
    test_loss, test_correct = validate(model, test_dataloader, bs)
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_correct / len(train_dataloader)
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_correct / len(test_dataloader)
    scheduler.step()
    print(f"Epoch: {epochs}")
    print(f"Train Error: Accuracy: {(100 * train_acc):>0.1f}%, Avg loss: {train_loss:>8f}\
            Test Error: Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

avg_train_loss = np.mean(history['train_loss'])
avg_test_loss = np.mean(history['test_loss'])
avg_train_acc = np.mean(history['train_acc'])
avg_test_acc = np.mean(history['test_acc'])
trained_folder = '"/kaggle/working/'
torch.save(model.state_dict(), trained_folder + f'/network.pt')

epochs = range(len(history['train_loss']))
plt.plot(epochs, history['train_loss'], history['test_loss'])
plt.legend(['train_loss', 'test_loss'])
plt.show()
plt.plot(epochs, history['train_acc'], history['test_acc'])
plt.legend(['train_acc', 'test_acc'])
plt.show()
