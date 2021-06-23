from typing import List
from numpy.lib.function_base import append
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.activation import RReLU
from torch.nn.modules.pooling import MaxPool2d
from dataLoading import *
import random

class CnnGene:
    def __init__(self, size : int, activationFuntion : int, maxPool : int):
        self.size = size
        self.activationFunction = activationFuntion
        self.maxPool = maxPool

class ActivationFuncType:
    ReLU = 0
    RReLU = 1
    ReLU6 = 2
    Max = 2

class MaxPoolType:
    NoMaxPool = 0
    MaxPool = 1
    Max = 1

class CnnDna:
    def __init__(self, imageChannels, imageRes, outputs, cnns : List[CnnGene], lnns):
        self.imageChannels = imageChannels
        self.imageRes = imageRes
        self.outputs = outputs
        self.cnns = cnns
        self.lnns = lnns

    def mutate(self):

        cnns = self.cnns.copy()
        lnns = self.lnns.copy()

        addCnn = 0.05
        removeCnn = 0.05
        incrementCnn = 0.5
        decrementCnn = 0.5
        flipActivation = 0.1
        flipMaxPool = 0.05

        rnd = random.random(0, 1)
        if (rnd < incrementCnn):
            random.choice(cnns).size *= 1.1

        rnd = random.random(0,1)
        if (rnd < decrementCnn):
            random.choice(cnns).size *= 0.9

        #rnd = random.random(0,1)
        #if (rnd < flipActivation)


        return CnnDna(self.imageChannels, self.imageRes, self.outputs, cnns, lnns)


    def createNetwork(self):
        modules = []
        i = 0
        # Add CNNs
        convDim = self.imageRes
        modules.append(nn.Conv2d(self.imageChannels, self.imageRes, kernel_size=3, padding=1))
        modules.append(nn.ReLU())
        num_prev = self.imageRes
        for cnn in self.cnns:
            modules.append(nn.Conv2d(num_prev, cnn.size, kernel_size=3, stride=1,padding=1))
            num_prev = cnn.size
            self.addActivationFunction(modules, cnn.activationFunction)
            self.addMaxPool(modules, cnn.maxPool)
            if (cnn.maxPool == MaxPoolType.MaxPool):
                convDim = convDim / 2
            i += 1

        # Add LNNs
        modules.append(nn.Flatten())
        modules.append(nn.Linear(self.cnns[-1].size * int(convDim) * int(convDim), self.lnns[0]))
        modules.append(nn.RReLU())
        i = 0
        for lnn in self.lnns:
            if (i == 0):
                i += 1
                continue

            modules.append(nn.Linear(self.lnns[i -1], lnn))
            modules.append(nn.ReLU())
            i += 1

        modules.append(nn.Linear(self.lnns[-1], self.outputs))

        network = nn.Sequential(*modules)
        to_device(network)
        return network

    def addMaxPool(self, modules : list, type):
        if (type == MaxPoolType.NoMaxPool):
            return False
        if (type == MaxPoolType.MaxPool):
            modules.append(nn.MaxPool2d(2,2))
            return True
        else:
            raise Exception("Unknown type " + str(type))

    def addActivationFunction(self, modules : list, type):
        if (type == ActivationFuncType.ReLU):
            modules.append(nn.ReLU())
        elif (type == ActivationFuncType.RReLU):
            modules.append(nn.RReLU())
        elif (type == ActivationFuncType.ReLU6):
            modules.append(nn.ReLU6())
        else:
            raise Exception("Unknown type " + str(type))

    def maxCnns(self):
        result = 0
        buffer = self.imageRes / 2

        while True:
            buffer = buffer / 2
            result += 1
            if (buffer <= 2):
                break

        return result

    def maxLnns(self):
        return 10