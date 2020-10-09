from pytorch_lightning import Trainer
# from RSModule_base import ResNet18 as ResNet18_base
# from RSModule_small import ResNet18 as ResNet18_small
# from RNNModule import LSTM
# from BiLstmAttModule import BiLstmAttention
# from CNNModule import CnnNet
from .DECbanModule import DECban
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from test_tube import Experiment, HyperOptArgumentParser
import torch
from pytorch_lightning import Trainer
import os

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

if __name__ == '__main__':
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    proteins = [fn.split('_')[0] for fn in os.listdir('../prep/npz_file/')]
    list.sort(proteins, key=lambda x: x)
    embed_name = ['7merR', '5merP', '5merP-7merR']
    model_name = 'DECban'
    for name in embed_name:
        if name == '7merR':
            inp_size = 100
        elif name == '5merP':
            inp_size = 128
        else:
            inp_size = 228
        for i, protein in enumerate(proteins):
            exp = Experiment(
                save_dir='../metric/board/'+protein+'/{}-{}/'.format(model_name, name)
            )
            checkpoint_callback = ModelCheckpoint(
                filepath='../metric/checkpoint/'+protein+'/{}-{}/weights.ckpt'.format(model_name, name),
                save_best_only=True,
                verbose=True,
                monitor='val_acc',
                mode='max'
            )
            early_stop = EarlyStopping(
                monitor='val_acc',
                patience=20
            )
            model = DECban(
                input_size       = inp_size,
                length           = 1000,
                rna_embed_fn     = '../prep/7MerRna100.json.npy',
                protein_embed_fn = '../prep/5MerProtein128.json.npy',
                data_fn          = '../prep/npz_file/'+protein+'_data.npz',
                embed_name       = name,
            )
            trainer = Trainer(
                gpus=[0],
                checkpoint_callback=checkpoint_callback,
                experiment=exp,
                add_log_row_interval=100,
                val_percent_check=1.0,
                max_nb_epochs=25,
                early_stop_callback=early_stop,
                gradient_clip=1.,
            )
            trainer.fit(model)
