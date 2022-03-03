#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as aa
from sentence_rep.bow import*
from sentence_rep.biLSTM import*
from classifier import*

class Model(nn.Module):
    def __init__(self,model, pre_train, freeze=True, pre_train_weight=0,vocab_size=0, emb_dim=50):
        super().__init__()

        if model == 'bow':
            if pre_train==False:
                pre_train_weight=0
            self.sentence_rep = BoW(pre_train_weight, pre_train, freeze)
        if model == 'bilstm':
            self.sentence_rep = BiLSTM(pre_train_weight, )
        self.classifier = Classifier(n_input=300, n_hidden=128, n_output=50)

    def forward(self,input):
        out1 = self.sentence_rep(input)
        out = self.classifier(out1)

        return out