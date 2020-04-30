import numpy as np
import argparse
import json

coden_dict = {'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',                             # alanine<A>
              'UGU': 'C', 'UGC': 'C',                                                 # systeine<C>
              'GAU': 'D', 'GAC': 'D',                                                 # aspartic acid<D>
              'GAA': 'E', 'GAG': 'E',                                                 # glutamic acid<E>
              'UUU': 'F', 'UUC': 'F',                                                 # phenylanaline<F>
              'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',                             # glycine<G>
              'CAU': 'H', 'CAC': 'H',                                                 # histidine<H>
              'AUU': 'I', 'AUC': 'I', 'AUA': 'I',                                       # isoleucine<I>
              'AAA': 'K', 'AAG': 'K',                                                 # lycine<K>
              'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',         # leucine<L>
              'AUG': 'M',                                                          # methionine<M>
              'AAU': 'N', 'AAC': 'N',                                               # asparagine<N>
              'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',                         # proline<P>
              'CAA': 'Q', 'CAG': 'Q',                                               # glutamine<Q>
              'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',   # arginine<R>
              'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',   # serine<S>
              'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',                         # threonine<T>
              'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',                         # valine<V>
              'UGG': 'W',                                                          # tryptophan<W>
              'UAU': 'Y', 'UAC': 'Y',                                               # tyrosine(Y)
              'UAA': 'Z', 'UAG': 'Z', 'UGA': 'Z',                                    # STOP code
              }


def gen_np_embedding(fn, word_idx_fn, out_fn, dim=100):
    with open(word_idx_fn) as f:
        word_idx = json.load(f)
    embedding=np.zeros((len(word_idx)+1, dim))
    with open(fn) as f:
        # protein embedding dictionary
        dic = json.load(f)  
    # k is 7mer RNA sequence 
    for k in word_idx.keys():
        if k == '<unk>':
            continue
        # 5mer Protein sequence
        protein = rna_to_protein(k)
        if 'Z' not in protein:
            protein_emb = []
            for p in protein:
                protein_emb.append(dic[p])
            # print(protein_emb)
            protein_emb = np.array(protein_emb, dtype='float')
            embedding[word_idx[k]] = np.resize(protein_emb, (640))
    # count = np.sum(np.sum(embedding, axis=1) == 0.)
    # print('the number of 7mer rna which contains stop code is {}'.format(count))
    with open(out_fn+'.oov.txt', 'w') as f:
        count = 0
        for w in word_idx:
            if embedding[word_idx[w]].sum() == 0. and len(w) == 7:
                count += 1
                f.write(w+'\n')
        print('the number of 7mer rna which contains stop code is {}'.format(count))
    np.save(out_fn+".npy", embedding.astype('float32'))


def rna_to_protein(rna_seq):
    protein = ""
    for idx in range(len(rna_seq)-2):
        protein += coden_dict[rna_seq[idx: idx+3].replace('T', 'U')]
    return protein


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', type=str, default='../embedding/')
    parser.add_argument('--out_dir', type=str, default='../prep_data/')
    parser.add_argument('--protein_emb', type=str, default="5MerProtein128.json")
    parser.add_argument('--word_idx', type=str, default="word_idx.json")
    parser.add_argument('--protein_dim', type=int, default=640)
    args = parser.parse_args()

    gen_np_embedding(args.emb_dir+args.protein_emb, args.out_dir+args.word_idx, args.out_dir+args.protein_emb, args.protein_dim)


if __name__ == '__main__':
    main()
    # protein_emb = np.load('../prep_data/1MerProtein128.json.npy')
    # print(protein_emb.shape)
    # l = [[0., 0., 0.], [0., 0., 0.], [0., 1., 0.], [1., 1., 0.]]
    # l = np.asarray(l)
    # print(l)
    # print(np.sum(l, axis=1))
    # print(np.sum(l, axis=1) == 0.)
    # print(np.sum(np.sum(l, axis=1) == 0.))