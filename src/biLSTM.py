#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.word_embeddings = nn.Embedding.from_pretrained(pre_train_weight,freeze=freeze)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim_bilstm, bidirectional=True)

    def delete_padding(self,sentence):
        sent = []
        for w in sentence[0]:
            if w != 0:
                sent.append(w)

        sent = torch.LongTensor(sent).view(1, -1)
        return sent

    def forward(self, sentence):
        batch_size = sentence.shape[0]

        sentence = self.delete_padding(sentence)

        embeds = self.word_embeddings(sentence)

        bilstm_out,_= self.bilstm(embeds.view(len(sentence[0]),batch_size,-1))
        back = bilstm_out[0, :, self.hidden_dim_bilstm:]
        forward = bilstm_out[len(embeds[0]) - 1, :, :self.hidden_dim_bilstm]

        out = torch.cat((forward,back), dim=0).view(embeds.shape[0], -1)
        return out