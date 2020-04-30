import numpy as np
import pandas as pd
import json
import os


class MakeData():
    def __init__(self, raw_data_fn, word_idx_fn):
        with open(word_idx_fn) as f:
            self.word_idx = json.load(f)
        frame         = pd.read_csv(raw_data_fn)
        self.data     = list(frame.iloc[:, 1])
        self.label    = list(frame.iloc[:, 2])
    
    def __str_to_idx(self, str_seq):
        rtn = np.zeros((1000,))
        maxLen = len(str_seq)
        for idx in range(maxLen//7):
            rtn[idx] = self.word_idx[str_seq[idx*7: (idx+1)*7]]
        return rtn.astype('int')

    def to_npz(self, out_fn):
        newdata     = [self.__str_to_idx(s) for s in self.data]
        self.data   = np.asarray(newdata)
        self.label  = np.asarray(self.label)
        # length      = len(self.data)
        # valid_split = np.floor(length*0.3)
        np.savez(out_fn, train_X=self.data, train_y=self.label)


if __name__ == '__main__':
    csvs = os.listdir('../prep_data/csv_file')
    list.sort(csvs, key=lambda x: x)
    for csv in csvs:
        makedata = MakeData('../prep_data/csv_file/'+csv, '../prep_data/word_idx.json')
        makedata.to_npz('../prep_data/npz_file/'+csv.split('.')[0]+'.npz')

    
    # data        = np.load('../prep_data/AGO1_data.npz')
    # valid_split = np.floor(len(data['train_y'])*0.3).astype('int')
    # valid_X     = data['train_X'][-valid_split:]
    # valid_y     = data['train_y'][-valid_split:]
    # print(valid_X)
    # print(valid_y)
    # train_X     = data['train_X'][:-valid_split]
    # train_y     = data['train_y'][:-valid_split]
    # print(train_X)
    # print(train_y)
