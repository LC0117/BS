from cProfile import label
import os
from unittest import result
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
from torch.utils.data import DataLoader as Dataloader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from datetime import datetime
import math
RESULTS = os.path.join(os.getcwd(), "results", "BS-LINEAR")

epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 3
torch.manual_seed(random_seed)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
act_coeff = 1  # controling sigmoid

dataset_train = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
dataset_test = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

data_train = Dataloader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
data_test = Dataloader(dataset=dataset_test, batch_size=batch_size_test, shuffle=True)


class BSActicvation(Function):
    @staticmethod
    def forward(ctx: torch.Any, input):
        result = 1/(1+(-4*input).exp())
        ctx.save_for_backward(result)
        # result = input.sigmoid()
        return result.bernoulli()

    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Any) -> torch.Any:
        result, = ctx.saved_tensors
        result = grad_output.sgn() * (4*(result * (1 - result))).bernoulli()
        return result


class BSLinear(nn.Module):
    def __init__(self,):
        super(BSLinear, self).__init__()
        self.l1 = nn.Linear(28*28, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 10)
        self.bs_activation = BSActicvation.apply
        # print(self.l3.weight.data)
        # print(self.l3.weight.size())
        # 初始化weights和bias
        # with torch.no_grad():
        #     self.l1.weight.data = 2*math.sqrt(512)*torch.randn(512, 28*28)
        #     self.l1.bias.data.fill_(0)
        #     self.l2.weight.data = 2*math.sqrt(256)*torch.randn(256, 512)
        #     self.l2.bias.data.fill_(0)
        #     self.l3.weight.data = 2*math.sqrt(10)*torch.randn(10, 256)
        #     self.l3.bias.data.fill_(0)


    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.bs_activation(self.l1(x))
        x = self.bs_activation(self.l2(x))
        return self.l3(x)


class SignedLoss(nn.Module):
    def forward(ctx, input, target):
        # log_softmax_output = F.log_softmax(output, dim=1)
        _, x = torch.max(input, dim=1)
        ctx.save_for_backward(x)
        signed_error = torch.sign(x - target)
        # cross_entropy = F.cross_entropy(input, target)
        # signed_error = torch.sign(cross_entropy)
        return signed_error
    
    # def backward(ctx, grad_output):
    #     grad_input = grad_output*torch.sign(ctx.saved_tensors[0])
    #     return grad_input


class SignedSGD(optim.SGD):
    def __init__(self, params, lr=0.1, batch_size=64) -> None:
        super(SignedSGD, self).__init__(params, lr)
        self.batch_size = batch_size

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']: 
                if p.grad is None:
                    continue
                p.data -= group['lr'] * p.grad.data/self.batch_size

        
def train_step(model, data_train, criterion, optimizer):
    total_loss = 0
    for images, labels in data_train:
        optimizer.zero_grad()
        output = model(images)
        # print("Output size")
        # print(output.size())
        # print("labels size")
        # print(labels.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(data_train.dataset)
    return model, optimizer, epoch_loss


def test_eval(data_test, model, criterion):
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in data_test:
        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    epoch_loss = total_loss / len(data_test.dataset)
    return model, epoch_loss, error, accuracy


def training_loop(model, criterion, optimizer, data_train, data_test, epochs):
    train_losses = []
    valid_losses = []
    test_error = []

    for epoch in range(0, epochs):
        model, optimizer, train_loss = train_step(model, data_train, criterion, optimizer)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, valid_loss, error, accuracy = test_eval(data_test, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Accuracy: {accuracy:.2f}%\t"
            )
    np.savetxt(os.path.join(RESULTS, "Test_error.csv"), test_error, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Train_Losses.csv"), train_losses, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Valid_Losses.csv"), valid_losses, delimiter=",")
    plot_results(train_losses, valid_losses, test_error)
    return model, optimizer, (train_losses, valid_losses, test_error)


def plot_results(train_losses, valid_losses, test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, "r-s", valid_losses, "b-o")
    plt.title("BS-Linear")
    plt.legend(fig[:2], ["Training Losses", "Validation Losses"])
    plt.xlabel("Epoch number")
    plt.ylabel("Loss [A.U.]")
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_losses.png"))
    plt.close()

    fig = plt.plot(test_error, "r-s")
    plt.title("BS-Linear")
    plt.legend(fig[:1], ["Validation Error"])
    plt.xlabel("Epoch number")
    plt.ylabel("Test Error [%]")
    plt.yscale("log")
    plt.ylim((5e-1, 1e2))
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_error.png"))
    plt.close()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # criterion = nn.CrossEntropyLoss()
    criterion = SignedLoss()
    model = BSLinear()
    optimizer = SignedSGD(model.parameters(), lr=0.1, batch_size=64)

    model, optimizer, _ = training_loop(model, criterion, optimizer, data_train, data_test, epochs)


if __name__ == "__main__":
    main()
