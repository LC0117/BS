import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
import torch.nn.functional as F

# Define the custom SignedCrossEntropyLoss
class SignedCrossEntropyLoss(nn.Module):
    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduction='none')
        signed_loss = torch.sign(loss) * loss
        return signed_loss.mean()

    def backward(self, grad_output):
        input, target = self.forward.saved_tensors
        grad_input = grad_output * torch.sign(F.cross_entropy(input, target, reduction='none'))
        return grad_input

# Define the CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        return self.fc3(x)


transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True,)

# Create the model, loss function, and optimizer
model = ConvNet()
criterion = SignedCrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')