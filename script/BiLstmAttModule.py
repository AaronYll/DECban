import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from Dataset import SeqData

import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score


class BiLstmAttmodule(nn.Module):
    def __init__(self, vocab_size=None, emb_dim=None,
                 lstm_dim=128, lstm_n_layer=2, lstm_dropout=0.3,
                 bidirectional=True, lstm_combine='add',
                 n_linear=2, linear_dropout=0.5, n_classes=2,
                ):
        super().__init__()
        n_dirs = bidirectional + 1
        lstm_dir_dim = lstm_dim // n_dirs if lstm_combine == 'concat' else lstm_dim

        self.lstm_n_layer = lstm_n_layer
        self.n_dirs = n_dirs
        self.n_dirs = n_dirs
        self.lstm_dir_dim = lstm_dir_dim
        self.lstm_combine = lstm_combine


        self.lstm = nn.LSTM(emb_dim, lstm_dir_dim,
                            num_layers=lstm_n_layer,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.att_w = torch.Tensor(torch.randn(1, lstm_dim, 1)).cuda()
        self.linear_layers = [nn.Linear(lstm_dim, lstm_dim) for _ in
                              range(n_linear - 1)]
        self.linear_layers = nn.ModuleList(self.linear_layers)

        self.label = nn.Linear(lstm_dim, n_classes)

        self.opts = {
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'lstm_dim': lstm_dim,
            'lstm_n_layer': lstm_n_layer,
            'lstm_combine': lstm_combine,
            'n_linear': n_linear,
            'n_classes': n_classes,
        }

    def re_attention(self, lstm_output, final_h, input):
        batch_size, seq_len, _ = input.shape

        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size,
                               self.lstm_dir_dim)[-1]
        final_h = final_h.permute(1, 0, 2)
        final_h = final_h.sum(dim=1)  # (batch_size, 1, self.half_dim)

        if self.lstm_combine == 'add':
            lstm_output = lstm_output.view(batch_size, seq_len, 2,
                                           self.lstm_dir_dim)
            lstm_output = lstm_output.sum(dim=2)
        att = torch.bmm(torch.tanh(lstm_output),
                        self.att_w.repeat(batch_size, 1, 1))
        att = F.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, input):
        batch_size, seq_len, *_ = input.shape

        lstm_output, (final_h, final_c) = self.lstm(input)

        attn_output = self.re_attention(lstm_output, final_h, input)
        output = attn_output

        for layer in self.linear_layers:
            output = layer(output)
            output = F.relu(output)

        logits = self.label(output)
        return logits


def bilstmattention(input_size=None, length=None):
    return BiLstmAttmodule(vocab_size=length, emb_dim=input_size)


class BiLstmAttention(pl.LightningModule):
    
    def __init__(self, input_size=None, length=None, lr=0.001, rna_embed_fn=None, protein_embed_fn=None, data_fn=None, embed_name=None):
        super(BiLstmAttention, self).__init__()
        self.bilstmattn = bilstmattention(input_size=input_size, length=length)

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
        return self.bilstmattn(x)
    
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


