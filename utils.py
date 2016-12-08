import pandas as pd
import numpy as np
import os
import tqdm
from sklearn.cross_validation import train_test_split


def store(path, history):
    with open(path, 'a') as fout:
        for line in history:
            fout.write('{}\t{}\n'.format(*line))


def read_fa(path):
    # TODO: check format
    # TODO: ask about names
    with open(path, 'r') as fin:
        text = (_ for _ in fin.read().split('>') if len(_) > 0)
        ret = [t[t.index('\n') + 1:].replace('\n', '') for t in text]
        return ret


def encode(seq):
    # TODO: what is N?
    vocab = {t: i for i, t in enumerate('NATGC')}
    ret = np.zeros((len(seq)), dtype=np.int8)
    ret[:] = [vocab[t] for t in seq]
    return ret


def prepare_data(min_len, path='./data', seed=1337):
    sets = []
    for p in ['gencode.v19.pc_transcripts.fa', 'lncipedia_3_1.fasta']:
        data = read_fa(os.path.join(path, p))
        sets.append([encode(t) for t in data if len(t) > min_len])

    gen, lnc = sets

    data = gen + lnc
    labels = [1] * len(gen) + [0] * len(lnc)

    # let's do a little shuffle
    idx = np.arange(len(labels))
    np.random.seed(seed)
    np.random.shuffle(idx)

    data = [data[_] for _ in idx]
    labels = [labels[_] for _ in idx]

    # data is a list of lists with different len
    # labels is just a {0, 1} labels
    return (data, labels)


def quick_test():
    data, labels = prepare_data(128)
    print(len(data))
    print(data[0])


if __name__ == "__main__":
    quick_test()
#
#
# def read_FASTA(file_name):
#     with open(file_name, 'r') as fin:
#         text = fin.read().split(">")
#         text = [x.split("\n") for x in text if x != ""]
#         text = [[x[0].split("|")[0],"".join(x[1:])] for x in text]
#         return text
#
#
# def encode_seq(seq):
#     # we want int-encoded sample: [0,1,2,3,2,2,2,2,1,1,]
#     res = np.zeros((len(seq),), dtype=np.int8)
#     map_dic = {'A': 1, 'T': 2, 'G': 3, 'C': 4}
#     i = 0
#     for i, b in enumerate(seq):
#         if b == 'N':
#             continue
#         res[i] = map_dic[b]
#     return res
#
#
# def process_data(seq_array, thresh = 10000):
#     # TODO: refactor it
#     g_len = np.array([len(x[1]) for x in seq_array])
#     idx = np.arange(len(seq_array))[g_len<=thresh]
#     tmp = [seq_array[i] for i in idx]
#     result = [encode_seq(x[1]) for x in tmp]
#     names = [x[0] for x in tmp]
#     return result, names
#
#
# def load_data():
#     gencode = read_FASTA("gencode.v19.pc_transcripts.fa")
#     lnc = read_FASTA("lncipedia_3_1.fasta")
#
#     data_encoded_gencode, names_encoded_gencode = process_data(gencode)
#     data_encoded_lnc, names_encoded_lnc = process_data(lnc)
#     data_encoded_gencode.extend(data_encoded_lnc[:len(data_encoded_gencode)])
#     targets = [1] * 94502 + [0] * 94502
#
#
#     # modify here
#     ret = train_test_split(data_encoded_gencode, targets, test_size=0.3, stratify=targets)
#     return [np.array(_) for _ in ret]
#
