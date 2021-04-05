import torch
import torchvision.models as models
import torch.nn as nn
from torch.optim import SGD, optimizer 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset_ants_bees import train_dataset, val_dataset
from src.dataset_ants_bees import train_loader, val_loader

import numpy as np
from sklearn.metrics import accuracy_score


# Define NN
net = models.AlexNet(pre_trained=True)


if __name__ == '__main__':
    print(net)