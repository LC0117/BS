import os
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
from torch.utils.data import DataLoader as Dataloader
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime

RESULTS = os.path.join(os.getcwd(), "results", "LENET5")

epochs = 50
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 3
torch.manual_seed(random_seed)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_train = torchvision.datasets.MNIST(root='./data/',train=True,download=True,transform=transform)
data_test = torchvision.datasets.MNIST(root='./data/',train=False,download=True,transform=transform)

data_train = Dataloader(dataset=data_train,batch_size=batch_size_train,shuffle=True)
data_test = Dataloader(dataset=data_test,batch_size=batch_size_test,shuffle=True)


class BSActicvation(Function):
    @staticmethod
    def forward(ctx: torch.Any, input):
        ctx.save_for_backward(input)
        result = input.sigmoid().bernoulli()
        return result

    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Any) -> torch.Any:
        input, = ctx.saved_tensors
        sigmoid_input = input.sigmoid()
        result = grad_output * (sigmoid_input * (1 - sigmoid_input)).bernoulli()
        # print(result)
        return result


class Lenet5(nn.Module):
    def __init__(self,):
        super(Lenet5, self).__init__()
        # 卷积层
        # self.block1 = nn.Sequential(
        #     # - out_channels: 卷积核的个数
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # 输出6*28*28
        #     BSActicvation.apply,
        #     nn.MaxPool2d(kernel_size=2, stride=2), # 输出6*14*14
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # 输出16*10*10
        #     BSActicvation.apply,
        #     nn.MaxPool2d(kernel_size=2, stride=2), # 输出16*5*5
        #     BSActicvation.apply
        # )
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bs_activation = BSActicvation.apply
        # 全连接层
        # self.block2 = nn.Sequential(
        #     nn.Linear(16*5*5, 120),
        #     BSActicvation.apply,
        #     nn.Linear(120, 84),
        #     BSActicvation.apply,
        #     nn.Linear(84, 10)
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bs_activation(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bs_activation(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bs_activation(x)
        x = self.fc2(x)
        x = self.bs_activation(x)
        x = self.fc3(x)
        return x


def train_step(model, data_train, criterion, optimizer):
    total_loss = 0
    for images, labels in data_train:
        optimizer.zero_grad()
        output = model(images)
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
    plt.title("LeNet5")
    plt.legend(fig[:2], ["Training Losses", "Validation Losses"])
    plt.xlabel("Epoch number")
    plt.ylabel("Loss [A.U.]")
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_losses.png"))
    plt.close()

    fig = plt.plot(test_error, "r-s")
    plt.title("LeNet5")
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
    criterion = nn.CrossEntropyLoss()
    model = Lenet5()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model, optimizer, _ = training_loop(model, criterion, optimizer, data_train, data_test, epochs)


if __name__ == "__main__":
    main()
