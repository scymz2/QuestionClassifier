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
                 hidden_dim):

        super().__init__()
        self.hidden_dim = hidden_dim

        if pre_train == True:
            self.word_embeddings = nn.Embedding.from_pretrained(pre_train_weight,freeze=freeze)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        return lstm_out
