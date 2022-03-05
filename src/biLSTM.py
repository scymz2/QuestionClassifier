#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self,
                 pre_train_weight,
                 vocab_size,
                 pre_train,
                 freeze,
                 embedding_dim,
                 hidden_dim_bilstm):
        super().__init__()
        self.hidden_dim_bilstm = hidden_dim_bilstm
        if pre_train == True:
            self.word_embeddings = nn.Embedding.from_pretrained(pre_train_weight,freeze=freeze, padding_idx=0)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # self.bilstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim_bilstm, bidirectional=True)
        # self.hidden2sent = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, sentence):
        batch_size = sentence.shape[0]
        embeds = self.word_embeddings(sentence)
        bilstm_out,_ = self.bilstm(embeds.view(len(sentence[0]),batch_size,-1))
        out = bilstm_out[-1]
        # out = self.hidden2sent(lstm_out.view())
        return out
