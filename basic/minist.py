from ast import mod
from cgi import test
from datetime import datetime
import os
from numpy import imag
from sympy import tanh
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader as Dataloader
from matplotlib import pyplot as plt
from torch import mode, nn
import torch.optim as optim

RESULTS = os.path.join(os.getcwd(), "results", "LENET5")

epochs = 50
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

data_train = torchvision.datasets.MNIST(root='./data/',train=True,download=True,transform=transform)
data_test = torchvision.datasets.MNIST(root='./data/',train=False,download=True,transform=transform)

data_train = Dataloader(dataset=data_train,batch_size=batch_size_train,shuffle=True)
data_test = Dataloader(dataset=data_test,batch_size=batch_size_test,shuffle=True)

# examples = enumerate(data_train)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data.shape)

# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

class Lenet(nn.Module):
    def __init__(self,):
        super(Lenet, self).__init__()
        # 卷积层
        self.block1 = nn.Sequential(
            # - out_channels: 卷积核的个数
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # 输出6*28*28
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # 输出16*10*10
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出16*5*5
            nn.Tanh()
        )
        # 全连接层
        self.block2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = x.view(-1, 16*5*5)
        x = self.block2(x)
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
    model = Lenet()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    model, optimizer, _ = training_loop(model, criterion, optimizer, data_train, data_test, epochs)


if __name__ == "__main__":
    main()
