import os
import torch 
import torchvision

from torchvision import datasets, transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
# building dataset
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir,'train',
    data_transforms['train']))
test_dataset = datasets.ImageFolder(
    os.path.join(data_dir,'test',
    data_transforms['test']))

# building dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=True)



if __name__ == '__main__':

    dataset_sizes = {
    'train':len(train_dataset),
    'test':len(test_dataset)
    }

    class_names = train_dataset.classes

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(class_names)
    # print(device)

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    print(classes)