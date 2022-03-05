import torch.nn as nn
import torch.nn.functional as F
import torch

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.ReLU()
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, sentence_rep):
        out = self.layer1(sentence_rep)
        out = self.layer2(out)
        out = self.predict(out)
        out = F.softmax(out,dim=0)
        # pred = out.max(1, keepdim=True)[1]
        return out
