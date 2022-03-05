import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.ReLU()
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, sentence_rep):
        out1 = self.layer1(sentence_rep)
        out2 = self.layer2(out1)
        out = self.predict(out2)
        out = nn.functional.softmax(out, dim=0)

        return out
