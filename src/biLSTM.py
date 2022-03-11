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

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim_bilstm, bidirectional=True)

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        bilstm_out, _ = self.bilstm(embeds.transpose(0,1))

        back = bilstm_out[0, :, self.hidden_dim_bilstm:]
        forward = bilstm_out[len(sentence[0]) - 1, :, :self.hidden_dim_bilstm]
        out = torch.cat((forward, back), dim=1)

        return out