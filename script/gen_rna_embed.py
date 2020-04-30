import numpy as np
import argparse
import json


def gen_np_embedding(fn, word_idx_fn, out_fn, dim=100):
    with open(word_idx_fn) as f:
        word_idx = json.load(f)
    embedding=np.zeros((len(word_idx)+1, dim))
    with open(fn) as f:
        dic = json.load(f)
    for k, v in dic.items():
        if k in word_idx:
            embedding[word_idx[k]] = np.array([float(x) for x in v])
    np.save(out_fn+".npy", embedding.astype('float32'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', type=str, default='../embedding/')
    parser.add_argument('--out_dir', type=str, default='../prep_data/')
    parser.add_argument('--rna_emb', type=str, default="7MerRna100.json")
    parser.add_argument('--word_idx', type=str, default="word_idx.json")
    parser.add_argument('--rna_dim', type=int, default=100)
    args = parser.parse_args()

    gen_np_embedding(args.emb_dir+args.rna_emb, args.out_dir+args.word_idx, args.out_dir+args.rna_emb, args.rna_dim)


if __name__ == '__main__':
    main()