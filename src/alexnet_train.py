import torch
import torchvision.models as models
import torch.nn as nn
from torch.optim import SGD, optimizer 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset_ants_bees import train_dataset, test_dataset
from src.dataset_ants_bees import train_loader, val_loader
from src.alexnet_model import MiniAlexNet

import numpy as np
from sklearn.metrics import accuracy_score
# Define NN
net = MiniAlexNet()

# define Optimizer
optimizer = SGD(
    net.parameters(),
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0005)

# Define Scheduler
scheduler = ReduceLROnPlateau(
    optimizer, 
    patience=4,
    factor=0.1, 
    mode = 'min')

# Define Criterion
criterion = nn.BCELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Model training
epochs = 1

losses_df = {
    'train_loss':[],
    'val_loss':[],
    'val_acc':[]
}
BATCH_SIZE = 4
num_batches = len(train_loader)//BATCH_SIZE
val_num_batches = len(val_loader)//BATCH_SIZE
best_loss = np.inf

for epoch in range(epochs):
    print('Epoch : ',epoch)
    scheduler.step(epoch)
    print('Learning rate :', get_lr(optimizer))
    print('Training for epoch {}'.format(epoch))
    val_outputs = []
    loss = 0
    vloss = 0

    net.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        img, label = batch
        img, label = img.to(device,dtype=torch.float), label.to(device,dtype=torch.long)
        output = net(img)
        batch_loss = criterion(output,label)

        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
    train_loss = loss/num_batches
    print('Training loss for epoch {} is {:.4f}'.format(epoch, train_loss))
    losses_df['train_loss'].append(train_loss)
    print('Validation for epoch {}'.format(epoch))

    with torch.no_grad():
        net.eval()
    
        for i, batch in enumerate(val_loader):
            img, label = batch
            img, label = img.to(device, dtype = torch.float), label.to(device, dtype = torch.long)
            output = net(img)
            batch_loss = criterion(output, torch.max(label, 1)[1])
            output = output.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            val_outputs.extend(np.argmax(output, 1))
            vloss += batch_loss.item()

    # val_loss = vloss/val_num_batches
    # print('Validation loss for epoch {} is {:.4f}'.format(epoch, val_loss))
    # losses_df['val_loss'].append(val_loss)
    # val_labels = list(np.argmax(np.array(val_y), 1))
    # acc = accuracy_score(val_outputs, val_labels)
    # print('Accuracy for epoch {} is {:.4f}'.format(epoch, acc))
    # losses_df['val_acc'].append(acc)
    
    # if val_loss <= best_loss:
    #     print('Validation loss has reduced from {:.4f} to {:.4f}'.format(best_loss, val_loss))
    #     print('Saving model')
    #     best_loss = val_loss
    #     torch.save(net.state_dict(), 'alexnet_finetuning.pth')



if __name__ == '__main__':
    print(net)