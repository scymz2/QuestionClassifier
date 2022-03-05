#!/usr/bin/python
# -*- coding: UTF-8 -*-

# !/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn


class BoW(nn.Module):
    def __init__(self,
                 pre_train_weight,
                 vocab_size,
                 pre_train,
                 freeze,
                 embedding_dim):

        super().__init__()
        if pre_train == True:
            self.bag_of_words = nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze, mode='mean', padding_idx=0)
        else:
            self.bag_of_words = nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean', padding_idx=0)

        # if pre_train == True:
        #     self.embedding = nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze, padding_idx=0)
        # else:
        #     self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, sentence):
        out = self.bag_of_words(sentence)
        return out
        # out1 = self.embedding(sentence)
        # out = torch.mean(out1, dim=1)
        # return out
