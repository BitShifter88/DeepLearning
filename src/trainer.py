import os
import torch
import torchvision
import tarfile
import genetic
import dataLoading
from dataLoading import DeviceDataLoader, get_default_device
from genetic import CnnDna
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self):
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, '.')
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        data_dir = './data/cifar10'
        classes = os.listdir(data_dir + "/train")
        dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

        random_seed = 42
        torch.manual_seed(random_seed)

        val_size = 5000
        train_size = len(dataset) - val_size

        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        batch_size=128

        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

        default_device = get_default_device()
        print(default_device)

        self.train_dl = DeviceDataLoader(train_dl)
        self.val_dl = DeviceDataLoader(val_dl)

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = self.evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    def evaluateDna(self, dna :CnnDna):
        model = Cifar10CnnModel(dna)
        lr = 0.001
        opt_func = torch.optim.Adam
        num_epochs = 10
        history = self.fit(num_epochs, lr, model, self.train_dl, self.val_dl, opt_func)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self, dna : CnnDna):
        super().__init__()
        self.network = dna.createNetwork()
    def forward(self, xb):
        return self.network(xb)
