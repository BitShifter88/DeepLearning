from trainer import *
from genetic import *

import matplotlib
import matplotlib.pyplot as plt

t = Trainer()

dna = CnnDna(0.001, 3, 32, 10, [
    CnnGene(64, ActivationFuncType.ReLU, MaxPoolType.MaxPool),
    CnnGene(128, ActivationFuncType.ReLU, MaxPoolType.NoMaxPool),
    CnnGene(128, ActivationFuncType.ReLU, MaxPoolType.MaxPool),
    CnnGene(256, ActivationFuncType.ReLU, MaxPoolType.NoMaxPool),
    CnnGene(256, ActivationFuncType.ReLU, MaxPoolType.MaxPool)
], [1024,512])

    # 64,
    # GeneType.ReLU,
    # GeneType.MaxPool,
    # 128,
    # GeneType.ReLU,
    # GeneType.NoMaxPool,
    # 128,
    # GeneType.ReLU,
    # GeneType.MaxPool,
    # 256,
    # GeneType.ReLU,
    # GeneType.MaxPool,
    # 256,
    # GeneType.ReLU,
    # GeneType.MaxPool

history = t.evaluateDna(dna)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


"""Our model reaches an accuracy of around 75%, and by looking at the graph, it seems unlikely that the model will achieve an accuracy higher than 80% even after training for a long time. This suggests that we might need to use a more powerful model to capture the relationship between the images and the labels more accurately. This can be done by adding more convolutional layers to our model, or incrasing the no. of channels in each convolutional layer, or by using regularization techniques.

We can also plot the training and validation losses to study the trend.
"""

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

plot_accuracies(history)
plot_losses(history)