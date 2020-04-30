import os
import json


def make_idx(embed_fn, out_fn):
    out = {}
    with open(embed_fn) as f:
        idx = 1
        # idx编码从1开始, 0保留给填充值
        dic = json.load(f)
        for k in dic.keys():
            if not k in out:
                out[k] = idx
                idx += 1
    with open(out_fn, 'w') as f:
        json.dump(out, f)


def make_dic(embed_fn, out_fn):
    out = {}
    with open(embed_fn) as f:
        for line in f:
            attribute = line.strip().split()
            name      = attribute[0]
            if not name in out:
                out[name] = [float(x) for x in attribute[1:]]
    with open(out_fn, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    make_idx('../embedding/7MerRna100.json', '../prep_data/word_idx.json')
    
    # 10merRna 对比实验
    # make_dic('../prep_data/10MerRna30.txt', '../prep_data/10MerRna30.json')
    # make_idx('../embedding/10MerRna30.json', '../prep_data/word_idx_10MerRna30.json')