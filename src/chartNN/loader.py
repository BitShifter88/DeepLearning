import torch
from torch.utils.data.dataset import Dataset
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
import os, os.path
import json

class ChartLoader(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.data = []
        self.LoadData()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = self.data[index]
        prices = data['Prices']
        prediction = data["Prediction"]
        tensor = torch.tensor(prices)
        predictionTensor = torch.tensor([prediction])
        return (tensor, predictionTensor)

    def LoadData(self):
        #counter = 0
        for filename in os.listdir(self.dir):
            with open(os.path.join(self.dir, filename)) as f:
                jsonData = json.load(f)
                self.data.append(jsonData)
                # counter += 1
                # if (counter == 100):
                #     break