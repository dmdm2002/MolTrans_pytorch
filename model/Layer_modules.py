from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class Embedding(nn.Module):
    """ Protein / Target embedding 생성, position embedding"""
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embedding, self).__init__()
        # 각 단어를 one-hot encoding 형식으로 변환
        # ex) A, B, C, D 라는 문자만 사용한다고 가정하고 [A, A, B] 일때 word embedding을 진행하면 [1, 1, 0, 0] 이 된다.
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        # 단어의 위치 정보를 주는 것
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_lengh = input_ids.size(1)

        position_ids = torch.arange(seq_lengh, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embedding = self.word_embedding(input_ids)
        position_embedding = self.position_embedding(position_ids)

        # 하나로 묶어주기 [word, position] 형태
        embeddings = words_embedding + position_embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# selfAttention 연산
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_ratio_dropout_ratio):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f'The hidden size {hidden_size} is not a multiplle of the number of attention head {num_attention_heads}'
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size/num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # query 생성
        self.q = nn.Linear(hidden_size, self.all_head_size)
        # key 생성
        self.k = nn.Linear(hidden_size, self.all_head_size)
        # value 생성
        self.v = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_ratio_dropout_ratio)

    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.q(hidden_states)
        mixed_key_layer = self.k(hidden_states)
        mixed_value_layer = self.v(hidden_states)

        q_layer = self.transpose_for_score(mixed_query_layer)
        k_layer = self.transpose_for_score(mixed_key_layer)
        v_layer = self.transpose_for_score(mixed_value_layer)

        # q, k 를 product 연산하여 raw attention score 를 얻는다.
        attention_scores = torch.matmul(q_layer, k_layer.traspose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Softmax 연산을 통해 attention probabilities 를 Normalize 한다.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # dropout한다.
        attention_probs = self.dropout(attention_probs)

        # context layer로 attention을 적용
        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


# selfAttention 결과 얻는 부분
class SelfAttention_Output(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfAttention_Output, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Multi Head Attention 정의
class Multi_Head_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_ratio_dropout_ratio, hidden_dropout_ratio):
        super(Multi_Head_Attention, self).__init__()
        self.SA = SelfAttention(hidden_size, num_attention_heads, attention_ratio_dropout_ratio)
        self.output = SelfAttention_Output(hidden_size, hidden_dropout_ratio)

    def forward(self, input_tensor, attention_mask):
        SA = self.SA(input_tensor, attention_mask)
        SA_output = self.output(SA, input_tensor)

        return SA_output


# transformer의 FFN 부분
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)

        return hidden_states


# 최종 output을 얻는 부분
class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_ratio):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# Transformer Encoder 생성
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_ratio_dropout_ratio, hidden_dropout_ratio):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(hidden_size, num_attention_heads, attention_ratio_dropout_ratio, hidden_dropout_ratio)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_ratio)

    def forward(self, hidden_sates, attention_mask):
        attention = self.attention(hidden_sates, attention_mask)
        intermediate_output = self.intermediate(attention)
        layer_output = self.output(intermediate_output)

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_ratio_dropout_ratio, hidden_dropout_ratio):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_ratio_dropout_ratio, hidden_dropout_ratio)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states