from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class SeqData(Dataset):
    def __init__(self, rna_embed_fn, protein_embed_fn, feature, label, embed_name):
        self.embed_name = embed_name
        self.rna_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(rna_embed_fn)
            )
        )
        self.protein_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(
                np.load(protein_embed_fn)
            )
        )
        self.feature = torch.from_numpy(feature).long()
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, item):
        feature = self.feature[item]
        # Double Embedding
        if self.embed_name == '5merP-7merR':
            newfeature = torch.cat(
                (self.rna_embedding(feature), self.protein_embedding(feature)),
                dim=1
            )
            label = self.label[item]

        # Single Embedding-Rna
        if self.embed_name == '7merR':
            newfeature = self.rna_embedding(feature)
            label = self.label[item]

        # Single Embedding-Protein-5Mer128
        if self.embed_name == '5merP':
            newfeature = self.protein_embedding(feature)
            label = self.label[item]

        return {
            'feature': newfeature,
            'label': label
        }


if __name__ == '__main__':
    data = np.load('../prep_data/AGO1_data.npz')
    trainset = SeqData('../prep_data/7MerRna100.json.npy', '../prep_data/1MerProtein128.json.npy', data['train_X'],
                       data['train_y'])
    tng_dataloader = DataLoader(trainset, batch_size=32)
    for i, batch in enumerate(tng_dataloader):
        feature = batch['feature']
        print(feature.size())
        break
