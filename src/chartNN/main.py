from loader import *
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from tqdm import tqdm
from torch.autograd import Variable
from model.multi_scale_ori import *
# %matplotlib inline

chartTest = ChartLoader("data/test")
chartTraining = ChartLoader("data/training")
chartValidation = ChartLoader("data/validation")

print(torch.seed())
#torch.manual_seed(10000)

# Use a white background for matplotlib figures
matplotlib.rcParams['figure.facecolor'] = '#ffffff'



val_size = 10000

batch_size=128

input_size = 2879
hidden_size = 32 # you can change this
num_classes = 3

testLoader = DataLoader(chartTest, batch_size, shuffle=True, num_workers=0, pin_memory=True)
trainingLoader = DataLoader(chartTraining, batch_size, shuffle=True, num_workers=0, pin_memory=True)
validationLoader =DataLoader(chartValidation, batch_size, shuffle=True, num_workers=0, pin_memory=True)





class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        l_size = 128
        self.linear1 = nn.Linear(in_size, l_size)
        self.linear2 = nn.Sequential(
            nn.Linear(l_size, l_size),
            nn.ReLU6(),
            nn.Linear(l_size,l_size),
            nn.ReLU6()
            )

        self.linear3 = nn.Sequential(
            nn.Linear(l_size, l_size),
            nn.ReLU6(),
            nn.Linear(l_size,l_size),
            nn.ReLU6()
            )

        self.linear5 = nn.Sequential(
            nn.Linear(l_size, l_size),
            nn.ReLU6(),
            nn.Linear(l_size,l_size),
            nn.ReLU6()
            )

        self.linear6 = nn.Sequential(
            nn.Linear(l_size, l_size),
            nn.ReLU6(),
            nn.Linear(l_size,l_size),
            nn.ReLU6()
            )

        # output layer
        self.linear4 = nn.Sequential(
            nn.Linear(l_size, out_size)
            )
        
    def forward(self, xb):
        # Flatten the image tensors
        #xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)

        out = self.linear2(out) + out
        out = self.linear3(out) + out
        out = self.linear5(out) + out
        out = self.linear6(out) + out

        # Get predictions using output layer
        out = self.linear4(out)
        #out = F.softmax(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        #print(images.shape)
        #print(images)
        out = self(images)    
        #out = torch.squeeze(labels)
        
        #print(out.shape)
        #print(labels.shape)                # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

"""We also need to define an `accuracy` function which calculates the accuracy of the model's prediction on an batch of inputs. It's used in `validation_step` above."""

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    # outputsF = torch.flatten(outputs)
    # labelsF = torch.flatten(labels)
    # diff = outputsF - labelsF
    # #_, preds = torch.max(outputs, dim=1)
    # # diff = torch.subtract(flat - labelsF)
    # # torch.set_printoptions(edgeitems=20)
    # # print(diff)
    # # print(diff.size)
    # # print(diff.shape)
    # diff = torch.abs(diff)
    # sum = torch.sum(diff)
    # length = len(outputsF)
    # avg = sum.item() / length 
    # #value = torch.sum(  torch.abs(outputsF - labels)).item() / len(outputsF)
    # #result = torch.tensor(value)
    
    # return torch.tensor(avg)

"""We'll create a model that contains a hidden layer with 32 activations."""

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

"""Let's take a look at the model's parameters. We expect to see one weight and bias matrix for each of the layers."""

for t in model.parameters():
    print(t.shape)

"""Let's try and generate some outputs using our model. We'll take the first batch of 128 images from our dataset and pass them into our model."""


"""## Using a GPU

As the sizes of our models and datasets increase, we need to use GPUs to train our models within a reasonable amount of time. GPUs contain hundreds of cores optimized for performing expensive matrix operations on floating-point numbers quickly, making them ideal for training deep neural networks. You can use GPUs for free on [Google Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/kernels) or rent GPU-powered machines on services like [Google Cloud Platform](https://cloud.google.com/gpu/), [Amazon Web Services](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html), and [Paperspace](https://www.paperspace.com/).

We can check if a GPU is available and the required NVIDIA CUDA drivers are installed using `torch.cuda.is_available`.
"""

torch.cuda.is_available()

"""Let's define a helper function to ensure that our code uses the GPU if available and defaults to using the CPU if it isn't. """

def get_default_device():
    return torch.device('cpu')

    #return torch.device('cpu')
    #return torch.device('cpu')
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
device

"""Next, let's define a function that can move data and model to a chosen device."""

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


"""Finally, we define a `DeviceDataLoader` class to wrap our existing data loaders and move batches of data to the selected device. Interestingly, we don't need to extend an existing class to create a PyTorch datal oader. All we need is an `__iter__` method to retrieve batches of data and an `__len__` method to get the number of batches."""

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

"""The `yield` keyword in Python is used to create a generator function that can be used within a `for` loop, as illustrated below."""

def some_numbers():
    yield 10
    yield 20
    yield 30

for value in some_numbers():
    print(value)

"""We can now wrap our data loaders using `DeviceDataLoader`."""

testLoader = DeviceDataLoader(testLoader, device)
trainingLoader = DeviceDataLoader(trainingLoader, device)
validationLoader = DeviceDataLoader(validationLoader, device)

"""Tensors moved to the GPU have a `device` property which includes that word `cuda`. Let's verify this by looking at a batch of data from `valid_dl`."""


"""## Training the Model

We'll define two functions: `fit` and `evaluate` to train the model using gradient descent and evaluate its performance on the validation set. For a detailed walkthrough of these functions, check out the [previous tutorial](https://jovian.ai/aakashns/03-logistic-regression).
"""

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

"""Before we train the model, we need to ensure that the data and the model's parameters (weights and biases) are on the same device (CPU or GPU). We can reuse the `to_device` function to move the model's parameters to the right device. """

# Model (on GPU)
model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
to_device(model, device)

result = evaluate(model, testLoader)
print(result)

"""Let's see how the model performs on the validation set with the initial set of weights and biases."""






msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=2)
msresnet = msresnet.cpu()

criterion = nn.CrossEntropyLoss(size_average=False).cpu()

optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)

num_epochs = 20

train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])
num_train_instances = len(trainingLoader)
num_test_instances = len(testLoader)

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    msresnet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    for (samples, labels) in tqdm(trainingLoader):
        samplesV = Variable(samples.cpu())
        labels = labels.squeeze()
        labelsV = Variable(labels.cpu())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        samplesV = torch.unsqueeze(samplesV, 2)
        samplesV = torch.movedim(samplesV, 2,1)
        predict_label = msresnet(samplesV)

        loss = criterion(predict_label[0], labelsV)

        loss_x += loss.item()

        loss.backward()
        optimizer.step()

    train_loss[epoch] = loss_x / num_train_instances

    msresnet.eval()
    # loss_x = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(trainingLoader):
        with torch.no_grad():
            samplesV = Variable(samples.cpu())
            labels = labels.squeeze()
            labelsV = Variable(labels.cpu())
            # labelsV = labelsV.view(-1)

            predict_label = msresnet(samplesV)
            prediction = predict_label[0].data.max(1)[1]
            correct_train += prediction.eq(labelsV.data.long()).sum()

            loss = criterion(predict_label[0], labelsV)
            # loss_x += loss.item()

    print("Training accuracy:", (100*float(correct_train)/num_train_instances))

    # train_loss[epoch] = loss_x / num_train_instances
    train_acc[epoch] = 100*float(correct_train)/num_train_instances

    trainacc = str(100*float(correct_train)/num_train_instances)[0:6]


    loss_x = 0
    correct_test = 0
    for i, (samples, labels) in enumerate(testLoader):
        with torch.no_grad():
            samplesV = Variable(samples.cpu())
            labels = labels.squeeze()
            labelsV = Variable(labels.cpu())
            # labelsV = labelsV.view(-1)

        predict_label = msresnet(samplesV)
        prediction = predict_label[0].data.max(1)[1]
        correct_test += prediction.eq(labelsV.data.long()).sum()

        loss = criterion(predict_label[0], labelsV)
        loss_x += loss.item()

    print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

    test_loss[epoch] = loss_x / num_test_instances
    test_acc[epoch] = 100 * float(correct_test) / num_test_instances

    testacc = str(100 * float(correct_test) / num_test_instances)[0:6]

    if epoch == 0:
        temp_test = correct_test
        temp_train = correct_train
    elif correct_test>temp_test:
        torch.save(msresnet, 'weights/changingResnet/ChaningSpeed_Train' + trainacc + 'Test' + testacc + '.pkl')
        temp_test = correct_test
        temp_train = correct_train












history = [evaluate(model, validationLoader)]
history


"""The initial accuracy is around 10%, as one might expect from a randomly initialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).

Let's train the model for five epochs and look at the results. We can use a relatively high learning rate of 0.5.
"""

history += fit(10, 0.0000001, model, trainingLoader, validationLoader)
history += fit(10,  0.0000001, model, trainingLoader, validationLoader)
history += fit(10, 0.00000001, model, trainingLoader, validationLoader)



#history += fit(5, 0.01, model, trainingLoader, validationLoader)
#history += fit(5, 0.001, model, trainingLoader, validationLoader)
#history += fit(10, 0.0001, model, trainingLoader, validationLoader)

print(torch.seed())

result = evaluate(model, testLoader)
print(result)

successCount = 0
failCount = 0
for x in range(1000):
    data, p = chartTest[x]
    #data = torch.unsqueeze(data, 0)
    data = to_device(data, get_default_device())
    pp = model(data)
    v1 = pp.cpu().detach().numpy()[0]
    v2 = pp.cpu().detach().numpy()[1]
    #v3 = pp.cpu().detach().numpy()[2]

    max = torch.max(torch.tensor([v1, v2]))
    #final = 0;

    #if (v2 > v1 and v2 > v3):
    #    final = 1
    #if (v3 > v1 and v3 > v2):
    #    final = 2

    if (pp.item() < 0 and p.item() < 0):
        successCount += 1
    elif (pp.item() > 0 and p.item() > 0):
        successCount += 1
    else:
        failCount += 1


    #if (v1 > v2):
    #    up = False

    print(str(pp.item()) + " - " + str(p.item()))
    #print(p)

print(str(successCount) + " - " + str(failCount))