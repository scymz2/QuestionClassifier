#!/usr/bin/python
# -*- coding: UTF-8 -*-

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn


class BoW(nn.Module):
    def __init__(self, pre_train_weight, vocab_size, emb_dim, pre_train, freeze, offset):
        super().__init__()

        if pre_train == True:
            self.bag_of_words = nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze, mode='mean')
        else:
            self.bag_of_words = nn.EmbeddingBag(vocab_size, emb_dim, freeze=False, mode='mean')

    def forward(self, x):
        output = self.bag_of_words(x)
        return output