# -*- coding: utf-8 -*-
"""cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o-EPqBTHVqVEMfvXDficqw8l_veQ7eWp
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class DNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, nn_depth, nn_width, dropout, momentum):
        super().__init__()

        self.bn_in = nn.BatchNorm1d(input_dim, momentum=momentum)
        self.dp_in = nn.Dropout(dropout)
        self.ln_in = nn.Linear(input_dim, nn_width, bias=False)

        self.bnorms = nn.ModuleList([nn.BatchNorm1d(nn_width, momentum=momentum) for i in range(nn_depth-1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(nn_depth-1)])
        self.linears = nn.ModuleList([nn.Linear(nn_width, nn_width, bias=False) for i in range(nn_depth-1)])
        
        self.bn_out = nn.BatchNorm1d(nn_width, momentum=momentum)
        self.dp_out = nn.Dropout(dropout/2)
        self.ln_out = nn.Linear(nn_width, output_dim, bias=False)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.bn_in(x)
        x = self.dp_in(x)
        x = nn.functional.relu(self.ln_in(x))

        for bn_layer,dp_layer,ln_layer in zip(self.bnorms,self.dropouts,self.linears):
            x = bn_layer(x)
            x = dp_layer(x)
            x = ln_layer(x)
            x = nn.functional.relu(x)
            
        x = self.bn_out(x)
        x = self.dp_out(x)
        x = self.ln_out(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        y_probs = torch.sigmoid(y_logit).detach().cpu().numpy()
        loss = self.loss(y_logit, y)
        metric = roc_auc_score(y.cpu().numpy(), y_probs)
        self.log('test_loss', loss)
        self.log('test_metric', metric)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]