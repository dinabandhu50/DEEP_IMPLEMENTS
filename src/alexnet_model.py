import torch
from torchvision.models import AlexNet
from src.dataset_ants_bees import train_dataset, test_dataset
from src.dataset_ants_bees import train_loader, test_loader

net = AlexNet()


if __name__ == '__main__':
    print(net)