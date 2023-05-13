# loosely based on https://github.com/pytorch/examples/blob/main/mnist/main.py

import argparse
import copy
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


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
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# stolen from https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/blob/master/LeNet-5%20Implementation%20Using%20Pytorch.ipynb
class LeNet(nn.Module):
    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

def save_params(model):
    params = []
    for param in model.parameters():
        params.append(param.clone().detach().cpu())
    return params


def set_params(model):
    model.load_state_dict(sd)


def save_grads(model):
    gradients = []
    for param in model.parameters():
        # print(param.grad.clone().detach().cpu().numpy().shape) ; exit()
        gradients.append(param.grad.clone().detach().cpu())
    return gradients


# for the i-th trainable parameter tensor t,
# grad_tables[i] has shape (iteration_count x t.shape)
def merge_grads(full_grad_save):
    grad_tables = []
    for i in range(len(full_grad_save[0])):
        grads = [full_grad_save[j][i].numpy() for j in range(len(full_grad_save))]
        grads = np.array(grads)
        print(i, grads.shape)
        grad_tables.append(grads)
    return grad_tables


def train(model, device, train_loader, optimizer, epoch):
    full_grad_save = []
    model.half()
    model.train()
    initial_state_dict = copy.deepcopy(model.state_dict())
    for batch_idx, (data, target) in enumerate(train_loader):
        # it should be 89% by now:
        # if batch_idx == 100: break
        data, target = data.to(device).half(), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        gradients = save_grads(model)
        full_grad_save.append(gradients)
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    final_state_dict = copy.deepcopy(model.state_dict())
    print(len(full_grad_save), "gradients collected")
    # grad_tables = merge_grads(full_grad_save)
    full_record = (initial_state_dict, final_state_dict, full_grad_save)
    return full_record


# almost the same as test(), should refactor
def evaluate(model, device, test_loader, n=1e9):
    model.eval()
    test_loss = 0
    correct = 0
    evaluated = 0
    with torch.no_grad():
        for data, target in test_loader:
            evaluated += len(data)
            data, target = data.to(device).half(), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if evaluated >= n:
                break

    test_loss /= evaluated
    test_acc = 100 * correct / evaluated
    return (test_loss, test_acc)


def apply_gradients(model, moment_state, gradients, lr, moment):
    device = "cuda"
    # print("moment", moment, "lr", lr) ; exit()
    for param_tensor, moment_tensor, gradient_tensor in zip(model.parameters(), moment_state, gradients):
        moment_tensor *= moment
        moment_tensor -= lr * gradient_tensor.to(device)
        param_tensor += moment_tensor


def replay(model, test_loader, initial_state_dict, final_state_dict, full_grad_save, lr, moment):
    device = "cuda"
    log_interval = 50
    model.half()
    model.train()
    model.load_state_dict(initial_state_dict)
    first_gradients = full_grad_save[0]
    moment_state = [torch.Tensor(np.zeros_like(param)).to(device) for param in first_gradients]
    with torch.no_grad():
        for batch_idx, gradients in enumerate(full_grad_save):
            apply_gradients(model, moment_state, gradients, lr, moment)
            if batch_idx % log_interval == 0:
                test_loss, test_acc = evaluate(model, device, test_loader)
                print(f"batch_idx: {batch_idx}\tloss: {test_loss:.6f}\tacc: {test_acc:.6f}")
    test_loss, test_acc = evaluate(model, device, test_loader)
    print(f"final\tloss: {test_loss:.6f}\tacc: {test_acc:.6f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).half(), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def create_loaders(train_kwargs, test_kwargs):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader


def pickle_filename(nick, bs, lr, moment):
    return f"{nick}.bs{bs}_lr{lr}_mom{moment}.pkl"




# this is where we set up everything:

batch_size = 64 # should be in sync with the way the pickle was built

lr = 0.08 ; moment = 0.0
# lr = 0.04 ; moment = 0.1
# lr = 0.02 ; moment = 0.5
# lr = 0.01 ; moment = 0.9

# nick = "record"
nick = "lenet"

# task = "train"
task = "replay"
do_shuffling = True

device = "cuda"


def main_train():
    torch.manual_seed(1)
    epochs = 1

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if device == "cuda":
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = create_loaders(train_kwargs, test_kwargs)

    model = LeNet().to(device)
    parameter_count = 0
    for param in model.parameters():
        print(tuple(param.shape))
        parameter_count += torch.numel(param)
    print("model parameter size", parameter_count)

    # optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=moment)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, epochs + 1):
        full_record = train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # scheduler.step()

    filename = pickle_filename("lenet", batch_size, lr, moment)
    print("saving gradient records to", filename)
    with open(filename, "wb") as f:
        pickle.dump(full_record, f)


def main_replay():
    filename = pickle_filename(nick, batch_size, lr, moment)

    # probably not worth loading to gpu, but cpu does not have float16.
    device = "cuda"

    if nick == "lenet":
        model = LeNet().to(device)
    elif nick == "record": # name for hysterical raisins.
        model = Net().to(device)

    with open(filename, "rb") as f:
        full_record = pickle.load(f)
    (initial_state_dict, final_state_dict, full_grad_save) = full_record
    print("full gradient record read from pickle", filename)

    '''
    first_grads = full_grad_save[0]
    for param in first_grads:
        print(torch.numel(param))
    exit()
    '''

    if do_shuffling:
        print("randomly shuffling gradients across minibatches before replay")
        random.shuffle(full_grad_save)

    train_kwargs = test_kwargs = {'batch_size': batch_size}
    train_loader, test_loader = create_loaders(train_kwargs, test_kwargs)

    print("replaying")
    replay(model, test_loader, initial_state_dict, final_state_dict, full_grad_save, lr, moment)


if __name__ == '__main__':
    if task == "train":
        main_train()
    elif task == "replay":
        main_replay()
    else:
        assert False, f"unknown task {task}"
