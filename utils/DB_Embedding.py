import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs


class DB_Encoder(data.Dataset):
    def __init__(self, root, list_IDs, labels, df_dti):
        super(DB_Encoder, self).__init__()
        self.root = root
        self.labels = labels
        self.list_IDs = list_IDs
        self.df_dti = df_dti

        bpe_file_names = ['protein_codes_uniprot.txt', 'drug_codes_chembl.txt']
        subword_units_maps = ['subword_units_map_uniprot.csv', 'subword_units_map_chembl.csv']

        self.pbpe = self.get_bpe(bpe_file_names[0])
        self.dbpe = self.get_bpe(bpe_file_names[1])

        self.idx2word_p, self.words2idx_p = self.word2idx(subword_units_maps[0])
        self.idx2word_d, self.words2idx_d = self.word2idx(subword_units_maps[1])

        self.max_p = 545
        self.max_d = 50

    def get_bpe(self, file):
        vocab_path = f'{self.root}/{file}.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        bpe = BPE(bpe_codes_protein, merges=-1, separator='')

        return bpe

    def word2idx(self, file):
        sub_csv = pd.read_csv(f'{self.root}/{file}.csv')

        # idx 순서로 word 를 저장한다.
        idx2word = sub_csv['index'].values
        # 각각의 word에 해당하는 index를 mapping 해서 dict 형태로 만들어 준다.
        words2idx = dict(zip(self.idx2word_p, range(0, len(self.idx2word_p))))

        return idx2word, words2idx

    def protein2emb_encoder(self, x):
        t1 = self.pbpe.process_line(x).split()

        try:
            i1 = np.asarray([self.words2idx_p[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)

        if l < self.max_p:
            i = np.pad(i1, (0, self.max_p - 1), 'constant', constant_values=0)
            input_mask = ([1] * 1) + ([0] * (self.max_p - 1))

        else:
            i = i1[:self.max_p]
            input_mask = [1] * self.max_p

        return i, np.asarray(input_mask)

    def drug2emb_encoder(self, x):
        max_d = 100
        t1 = self.dbpe.process_line(x).split()

        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])
            # print(x)

        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))

        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        idx = self.list_IDs[idx]
        d = self.df_dti.iloc[idx]['SMILES']
        p = self.df_dti.iloc[idx]['Target Sequence']

        d_v, input_mask_d = self.drug2emb_encoder(d)
        p_v, input_mask_p = self.protein2emb_encoder(p)

        y = self.labels[idx]

        return [d_v, p_v, input_mask_d, input_mask_p, y]

