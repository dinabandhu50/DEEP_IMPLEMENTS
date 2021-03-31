import torch
import torch.optim as optim

def train_net(model, epochs,dataoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for epoch in epochs:
        for x,y in dataoader:
            x = x.to(device)
            y = y.to(device)
