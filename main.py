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


def train(args, model, device, train_loader, optimizer, epoch):
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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
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


def pickle_filename(bs, lr, moment):
    return f"record.bs{bs}_lr{lr}_mom{moment}.pkl"


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default changed to 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = create_loaders(train_kwargs, test_kwargs)

    lr = 0.01
    moment = 0.9

    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=moment)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, args.epochs + 1):
        full_record = train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    filename = pickle_filename(args.batch_size, lr, moment)
    print("saving gradient records to", filename)
    with open(filename, "wb") as f:
        pickle.dump(full_record, f)


def main_replay():
    batch_size = 64 # should be in sync with the way the pickle was built
    # lr = 0.08 ; moment = 0.0
    # lr = 0.04 ; moment = 0.1
    # lr = 0.02 ; moment = 0.5
    lr = 0.01 ; moment = 0.9
    do_shuffling = True

    filename = pickle_filename(batch_size, lr, moment)

    # probably not worth loading to gpu, but cpu does not have float16.
    device = "cuda"
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
    # main() ; exit()
    main_replay() ; exit()

