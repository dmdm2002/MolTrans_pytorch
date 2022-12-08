from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from model.Layer_modules import *
from utils.Option import BIN_config_DBPE
import numpy as np

import collections
import math
import copy

torch.manual_seed(1)
np.random.seed(1)


class Interaction_Flat(nn.Module):
    def __init__(self, **config):
        super(Interaction_Flat, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']

        # densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        # encoder
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']

        self.flatten_dim = config['flat_dim']

        # specialized embedding with positional one
        self.demb = Embedding(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embedding(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)

        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)

        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )

    def forward(self, d, p, d_mask, p_mask):
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)

        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        d_emb = self.demb(d)  # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...

        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())

        # repeat to have the same tensor size for aggregation
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1)  # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1)  # repeat along drug size

        i = d_aug * p_aug  # interaction
        i_v = i.view(int(self.batch_size / self.gpus), -1, self.max_d, self.max_p)
        i_v = torch.sum(i_v, dim=1)
        i_v = torch.unsqueeze(i_v, 1)

        i_v = F.dropout(i_v, p=self.dropout_rate)

        f = self.icnn(i_v)

        f = f.view(int(self.batch_size / self.gpus), -1)

        score = self.decoder(f)

        return score