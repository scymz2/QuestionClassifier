#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np


class Pre_train_loader:
    def __init__(self):
        self.pretrain_weight = []
        self.pretrain_dict = {}

    def load_pretrain(self, path, vocab):
        """
        This function creates a dictionary for the words and vectors in glove.small.txt
        :param vocab: The vocabulary of the raw_data.txt
        :param path: The path of glove.small.txt
        :return: The dictionary
        """
        with open(path, 'r') as f:
            for line in f:
                line = line.split('\t')  # '#UNK#'  '-2.190000 -1.02137....'
                label = line[0].lower()  # '#UNK#'
                vector = line[1]  # '-2.19....'
                # create a dictionary for the words and vectors
                self.pretrain_dict[label] = vector  # key-value

        # only keep the words that appear in current text, others will be marked with #UNK#
        weight_unk = self.pretrain_dict['#unk#'].split(' ')
        weight_unk = [float(w) for w in weight_unk]

        self.pretrain_weight.append(list(np.zeros(len(weight_unk))))
        for word in vocab:
            if word in self.pretrain_dict:
                weight = self.pretrain_dict[word].split(' ')
                weight = [float(w) for w in weight]
                self.pretrain_weight.append(weight)
            else:
                self.pretrain_weight.append(weight_unk)
        return torch.FloatTensor(self.pretrain_weight)
