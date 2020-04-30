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


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear  = nn.Linear(32, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x.size   = (batch, 3, 32)
        # att_w.size = (batch, 3, 1)
        # print('Attention输入张量的维度: '+str(x.size()))
        att_w = torch.tanh(self.linear(x))
        att_w = self.softmax(att_w).transpose(1, 2)  # (batch, 1, 3)
        # print('Attention值的维度: '+str(att_w.size()))
        out   = torch.matmul(att_w, x)  # (batch, 1, 32)
        # print('注意力层输出张量维度: '+str(out.size()))
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size= None, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride)
        self.bn1   = nn.BatchNorm1d(out_dim)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DECbanModule(nn.Module):
    
    def __init__(self, input_size=None, length=None):
        super(DECbanModule, self).__init__()
        self.conv3 = nn.ModuleList()
        self.conv4 = nn.ModuleList()
        self.conv5 = nn.ModuleList()
        self.conv3.append(BasicBlock(input_size, 32, 3))
        self.conv3.extend([BasicBlock(32, 32, 3) for i in range(2)])
        self.conv4.append(BasicBlock(input_size, 32, 4))
        self.conv4.extend([BasicBlock(32, 32, 4) for i in range(2)])
        self.conv5.append(BasicBlock(input_size, 32, 5))
        self.conv5.extend([BasicBlock(32, 32, 5) for i in range(2)])
        self.Max3_pool = nn.ModuleList([nn.MaxPool1d(length-2*i) for i in range(1, 4)])
        self.Max4_pool = nn.ModuleList([nn.MaxPool1d(length-3*i) for i in range(1, 4)])
        self.Max5_pool = nn.ModuleList([nn.MaxPool1d(length-4*i) for i in range(1, 4)])
        self.linear    = nn.Linear(32*3, 2)
        
        self.att_1     = Attention()
        self.att_2     = Attention()
        self.att_3     = Attention()
    
    def forward(self, x):
        pool1 = []
        pool2 = []
        pool3 = []
        # conv-layer1
        x1_1 = self.conv3[0](x)
        x2_1 = self.conv4[0](x)
        x3_1 = self.conv5[0](x)
        pool1.append(self.Max3_pool[0](x1_1))   # (batch, 32, 1)
        pool1.append(self.Max4_pool[0](x2_1))   # (batch, 32, 1)
        pool1.append(self.Max5_pool[0](x3_1))   # (batch, 32, 1)
        pool1 = torch.stack(pool1, 2).squeeze()   # (batch, 32, 3)
        out_1 = self.att_1(pool1.transpose(1, 2).contiguous())   # (batch, 32)
        
        # conv-layer2
        x1_2 = self.conv3[1](x1_1)
        x2_2 = self.conv4[1](x2_1)
        x3_2 = self.conv5[1](x3_1)
        pool2.append(self.Max3_pool[1](x1_2))
        pool2.append(self.Max4_pool[1](x2_2))
        pool2.append(self.Max5_pool[1](x3_2))
        pool2 = torch.stack(pool2, 2).squeeze()
        out_2 = self.att_2(pool2.transpose(1, 2).contiguous())   # (batch, 32)

        # conv-layer3
        x1_3 = self.Max3_pool[2](self.conv3[2](x1_2))
        x2_3 = self.Max4_pool[2](self.conv4[2](x2_2))
        x3_3 = self.Max5_pool[2](self.conv5[2](x3_2))
        pool3.append(x1_3)
        pool3.append(x2_3)
        pool3.append(x3_3)
        pool3 = torch.stack(pool3, 2).squeeze()     # (batch, 32, 3)
        out_3 = self.att_3(pool3.transpose(1, 2).contiguous())   # (batch, 1, 32)

        # concat
        out = torch.cat((out_1, out_2, out_3), -1)  # (batch, 1, 96*2)
        out = out.view(out.size(0), -1)   # (batch, 96)
        out = self.linear(out)
        return out  # (batch, 2)


def decban(input_size=None, length=None):
    return DECbanModule(input_size, length)


class DECban(pl.LightningModule):

    def __init__(self, input_size=None, length=None, lr=0.001, rna_embed_fn=None, protein_embed_fn=None, data_fn=None, embed_name=None):
        super(DECban, self).__init__()
        # self.resnet = resnet18(input_size=input_size, length=length)
        self.nn = decban(input_size=input_size, length=length)

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
        return self.nn(x)
    
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
            'y_true'   : y,
            'y_pred'   : y_pred
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
