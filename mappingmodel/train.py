#!/usr/bin/env python
"""
Training/Validation Module
"""
from pathlib import Path
import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from data import create_dir

def l2_reg(params, device):
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += torch.norm(param, 2) ** 2
    return penalty


def loss(y_hat, y, params, device, smooth=0.2, weights=[0.6, 0.9, 0.2], lambda_reg=0.0005):
    penalty = l2_reg(params, device)
    dice = dice_loss(y_hat, y, device, weights, smooth)
    bce = bce_loss(y_hat, y, device, weights)
    return dice + bce + lambda_reg * penalty


def dice_loss(y_hat, y, device, weights, smooth):
    dims = (0,) + tuple(range(2, y.ndimension()))
    intersection = torch.sum(y_hat * y, dims)
    union = torch.sum(y_hat + y, dims)
    dice = 1 - (2. * intersection / (union + smooth))
    weights = torch.Tensor(weights).to(device)
    return (weights * dice).mean()


def bce_loss(y_hat, y, device, weights):
    w_mat = torch.ones_like(y).to(device)
    for k in range(y.shape[1]):
        w_mat[:, k, :, :] = weights[k]
    return F.binary_cross_entropy(y_hat, y, weight=w_mat, reduction="mean")


def dice_bce_loss(y_hat, y, device, smooth=0.2, weights=[0.6, 0.9, 0.2]):
    y_hat = y_hat.view(-1)
    y = y.view(-1)
    import pdb
    pdb.set_trace()

    intersection = (y_hat * y).sum()
    dice_loss = 1 - (2. * intersection + smooth)/(y_hat.sum() + y.sum() + smooth)
    BCE = F.binary_cross_entropy(y_hat, y, reduction="mean")
    return BCE + dice_loss


def train_epoch(model, loader, optimizer, device, epoch=0):
    loss_ = 0
    model.train()
    n = len(loader.dataset)
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # gradient step
        optimizer.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y, model.parameters(), device)
        l.backward()
        optimizer.step()

        # compute losses
        loss_ += l.item()
        log_batch(epoch, i, n, loss_, loader.batch_size)

    return loss_ / n


def validate(model, loader):
    loss = 0
    model.eval()
    batch_size = False
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            y_hat = model(x)
            loss += loss(y_hat, y)

    return loss / len(loader.dataset)


def log_batch(epoch, i, n, loss, batch_size):
    print(
        f"Epoch: {epoch}\tbatch: {i} of {int(n) // batch_size}\tEpoch loss: {loss/batch_size:.5f}",
        end="\r",
        flush=True
    )


def predictions(model, ds, out_dir, device):
    create_dir(out_dir)

    for i in range(len(ds)):
        x, y = ds[i]
        ix = re.search("[0-9]+", str(ds.x_paths[i])).group(0)
        print(f"saving {i + 1}/{len(ds)}...", end="\r", flush=True)

        with torch.no_grad():
            y_hat = model(x.unsqueeze(0).to(device))
            np.save(out_dir / f"y_hat-{ix}.npy", y_hat.cpu()[0])
            np.save(out_dir / f"y-{ix}.npy", y)
            np.save(out_dir / f"x-{ix}.npy", x)
