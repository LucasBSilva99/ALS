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

from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.metrics import accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger

class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride = 1, 
            padding=2,  
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

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


        y_real = y.cpu().numpy()
        y_pred = y_probs.round()

        # confusion_matrix = torchmetrics.functional.confusion_matrix(torch.tensor(y_pred).type(torch.int), y.type(torch.int), num_classes=2)

        # df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(2), columns=range(2))
        # plt.figure(figsize = (10,7))
        # fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        
        # self.logger.experiment.add_figure("Confusion matrix", fig_)

        self.log('test_loss', loss)
        self.log('test_metric', metric)
        self.log('accuracy', accuracy_score(y_real,y_pred))
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
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
    
def cross_train_test_cnn(X, y, k, epochs, input_size, hidden_size, num_classes, learning_rate, ref_df, fast_run, rand_state, gpu=0, batch_size = 64, early_stop = 20):

  gkf = StratifiedGroupKFold(n_splits = k, shuffle = True, random_state=rand_state)
  gkf.get_n_splits(X, y)

  scores = []

  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)

  X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
  
  for train_idx, test_idx in gkf.split(X_scaled, y, groups=ref_df['REF'].tolist()):

    X_train, X_test = X_scaled_df.iloc[list(train_idx)], X_scaled_df.iloc[list(test_idx)]
    y_train, y_test = y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    X_train_sm, y_train_sm = resample_data(X_train, y_train)

    dataset_train = AlsDataset(X_train_sm, y_train_sm)

    dataset_test = AlsDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
                      dataset_train, 
                      batch_size=batch_size,
                      shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                      dataset_test,
                      batch_size=batch_size,
                      shuffle=False)

    model = SoftOrdering1DCNN(
        input_dim=input_size,
        output_dim=num_classes, 
        sign_size=16, 
        cha_input=64, 
        cha_hidden=64, 
        K=2, 
        dropout_input=0.3, 
        dropout_hidden=0.3, 
        dropout_output=0.2
    )

    early_stop_callback = EarlyStopping(
      monitor='valid_loss',
      min_delta=.0,
      patience= early_stop,
      verbose=True,
      mode='min'
    )

    trainer = pl.Trainer(callbacks=[early_stop_callback], min_epochs=10, max_epochs=epochs, gpus=gpu, fast_dev_run = fast_run)

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_loader)
