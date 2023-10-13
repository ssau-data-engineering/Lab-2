import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
import glob
import os
import sys
from datetime import datetime

SAVE_MODEL_PATH = '/data/lab2_nn_train/models'
DATA_PATH = '/data/lab2_nn_train/data'
TEST_FILENAME = 'test_data.npz'
TRAIN_PREFIX = 'train_'


BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
LR_STEP = 0.7
CUDA = False
SEED = 2023
LOG_INTERVAL = 10
SAVE_MODEL = True


class MnistDataset(Dataset):

    def __init__(self, x, y, transform=None):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]

        if self.transform:
            x_sample = self.transform(x_sample)

        return x_sample, self.y[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = CUDA and torch.cuda.is_available()

    torch.manual_seed(SEED)
    exp_folder_path = os.path.join('/data/lab2_nn_train/res', str(datetime.now()))
    os.makedirs(exp_folder_path, exist_ok=True)
    sys.stdout = open(os.path.join(exp_folder_path, "print.log"), 'w')

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': BATCH_SIZE}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    x_train_list = []
    y_train_list = []

    for single_train_file in glob.glob(os.path.join(DATA_PATH, f'{TRAIN_PREFIX}*.npz')):
        single_train_loaded = np.load(single_train_file)
        x_train_list.append(single_train_loaded['x_train'])
        y_train_list.append(single_train_loaded['y_train'])
    
    x_train_np = np.concatenate(x_train_list, axis=0)
    y_train_np = np.concatenate(y_train_list, axis=0)
    dataset_train = MnistDataset(
        x=x_train_np, y=y_train_np, 
        transform=transform
    )

    test_loaded = np.load(os.path.join(DATA_PATH, TEST_FILENAME))
    dataset_test = MnistDataset(
        x=test_loaded['x_test'], y=test_loaded['y_test'],
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

    scheduler = StepLR(optimizer, step_size=1, gamma=LR_STEP)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(exp_folder_path, 'mnist_cnn.pt'))


if __name__ == '__main__':
    main()