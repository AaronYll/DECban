import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataset import SeqData

import numpy as np
import math as math
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score


class CnnNetModule(nn.Module):
    
    def __init__(self, input_size=228, length=1000, num_classes=2):
        super(CnnNetModule, self).__init__()
        self.conv1   = torch.nn.Conv1d(input_size, 32, 5, padding=2 )
        self.conv2   = torch.nn.Conv1d(32, 32, 5, padding=2)
        self.conv3   = torch.nn.Conv1d(32, 32, 5, padding=2)
        self.maxpool = nn.MaxPool1d(length) 
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cnnnet(input_size=None, length=None):
    """construct a cnn model
    """
    model = CnnNetModule(input_size=input_size, length=length)
    return model


class CnnNet(pl.LightningModule):

    def __init__(self, input_size=None, length=None, lr=0.001, rna_embed_fn=None, protein_embed_fn=None, data_fn=None, embed_name=None):
        super(CnnNet, self).__init__()
        self.cnnnet = cnnnet(input_size=input_size, length=length)

        data        = np.load(data_fn)
        valid_split = np.floor(len(data['train_X'])*0.3).astype('int')
        
        # print(valid_split)
        train_X     = data['train_X'][:-valid_split]
        train_y     = data['train_y'][:-valid_split]
        valid_X     = data['train_X'][-valid_split:]
        valid_y     = data['train_y'][-valid_split:]
        self.trainset   = SeqData(rna_embed_fn, protein_embed_fn, train_X, train_y, embed_name)
        self.valset     = SeqData(rna_embed_fn, protein_embed_fn, valid_X, valid_y, embed_name)
        self.learning_rate = lr
    
    def forward(self, x):
        # x.size() = [batch_size, time_size, input_size]
        x = x.transpose(1, 2).contiguous()
        return self.cnnnet(x)
    
    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]

    def training_step(self, data_batch, batch_nb):
        x, y     = data_batch['feature'].cuda(), data_batch['label'].cuda()

        y_hat = self.forward(x)
        val_loss = self.loss(y_hat, y)
        val_acc = (torch.argmax(y_hat, dim=1) == y).sum(dim=0)/(len(y)*1.0)

        return {
            'loss': val_loss
        }
    
    def validation_step(self, data_batch, batch_nb):
        x, y     = data_batch['feature'].cuda(), data_batch['label'].cuda()
        y_hat    = self.forward(x)
        y_pred   = y_hat.argmax(dim=1)
        val_loss = self.loss(y_hat, y)
        

        val_acc  = (y_pred == y).sum(dim=0).item() / (len(y) * 1.0)
        val_acc  = torch.tensor(val_acc)

        return {
            'val_loss': val_loss,
            'val_acc' : val_acc,
            'y_true'  : y,
            'y_pred'  : y_pred
        }
    
    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean  = 0
        y_true        = []
        y_pred        = []

        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean  += output['val_acc']
            y_true.extend(list(np.array(output['y_true'].cpu())))
            y_pred.extend(list(np.array(output['y_pred'].cpu())))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        val_loss_mean /= len(outputs)
        val_acc_mean  /= len(outputs)

        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        auc       = roc_auc_score(y_true, y_pred)

        return {
            'val_loss': val_loss_mean.item(),
            'val_acc' : val_acc_mean.item(),
            'precision': precision,
            'recall'  : recall,
            'f1'      : f1,
            'auc'     : auc
        }
    
    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.trainset, batch_size=128, shuffle=True)
    
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=256, shuffle=True)