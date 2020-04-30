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


def conv3x1(in_dim, out_dim, stride=1):
    "3x1 convolution with paddding"
    return nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_dim, out_dim, stride=1, downsampling=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(in_dim, out_dim, stride)
        self.bn1   = nn.BatchNorm1d(out_dim)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(out_dim, out_dim)
        self.bn2   = nn.BatchNorm1d(out_dim)
        self.downsampling = downsampling
        self.stride = stride

    def forward(self, x):
        # x.size() = [batch_size, input_size, time_size]
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsampling is not None:
            residual = self.downsampling(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    
    def __init__(self, block, layers, input_size=228, length=1000, num_classes=2):
        self.in_dim = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.f = lambda x: math.ceil(x/32-7+1)
        self.fc = nn.Linear(128*block.expansion*self.f(length), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, dim, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_dim != dim*block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_dim, dim*block.expansion,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(dim*block.expansion)
            )

        layers = []
        layers.append(block(self.in_dim, dim, stride, downsample))
        self.in_dim = dim*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_dim, dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    """construct a ResNet-18 model
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class ResNet18(pl.LightningModule):

    def __init__(self, input_size=None, length=None, lr=0.001, rna_embed_fn=None, protein_embed_fn=None, data_fn=None, embed_name=None):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(input_size=input_size, length=length)

        data        = np.load(data_fn)
        valid_split = np.floor(len(data['train_X'])*0.3).astype('int')
        
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
        return self.resnet(x)
    
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