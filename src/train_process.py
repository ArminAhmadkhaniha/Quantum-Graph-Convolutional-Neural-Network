import torch
import numpy as np
from tqdm import tqdm


def train(model, train_loader, A_norm, optimizer, criterion):
    device = torch.device("cpu")
    model.train()
    total_loss = 0
    train_losses = []
    for batch_x, batch_y in tqdm(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x, A_norm)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))
    return train_losses