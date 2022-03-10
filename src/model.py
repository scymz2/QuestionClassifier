#!/usr/bin/python
# -*- coding: UTF-8 -*-
from bow import*
from biLSTM import*
from classification import*

class Model(nn.Module):
    def __init__(self,
                 model,
                 pre_train,
                 freeze,
                 pre_train_weight,
                 vocab_size,
                 embedding_dim,
                 hidden_dim_bilstm,
                 n_input,
                 n_hidden,
                 n_output):

        super().__init__()

        if model == 'bow':
            if pre_train==False:
                pre_train_weight=None
            self.sentence_rep = BoW(pre_train_weight, vocab_size, pre_train, freeze, embedding_dim)
        if model == 'bilstm':
            if pre_train==False:
                pre_train_weight=None
            self.sentence_rep = BiLSTM(pre_train_weight, vocab_size, pre_train, freeze,  embedding_dim, hidden_dim_bilstm)
        self.classification = Classification(n_input,n_hidden,n_output)

    def forward(self,input):
        out = self.sentence_rep(input)
        out = self.classification(out)

        return out