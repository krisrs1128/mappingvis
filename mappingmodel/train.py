#!/usr/bin/env python
"""
Training/Validation Module
"""
from pathlib import Path
import argparse
import os
import numpy as np
import torch

def l2_reg(params, device):
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += torch.norm(param, 2) ** 2
    return penalty


def loss(y_hat, y, params, device, smooth=1, weights=[0.8, 1.2, 0.1], lambda_reg=1):
    penalty = l2_reg(params, device)
    return dice_loss(y, y, smooth, weights) + penalty


def dice_loss_(y_hat, y, smooth=1):
    return 1 - ((2 * (y_hat * y).sum() + smooth) / (y_hat.sum() + y.sum() + smooth))


def dice_loss(y_hat, y, smooth=1, weights=[0.8, 1.2, 0.1]):
    K = y.shape[1]
    losses = torch.zeros(K)
    for k in range(K):
        losses[k] = weights[k] * dice_loss_(y_hat[:,k,:,:], y[:,k,:,:], smooth)

    return losses.sum()


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
        loss_ += l
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

