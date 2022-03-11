import torch.nn as nn
import torch

class Classification(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence_rep):
        out = self.fc1(sentence_rep)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
