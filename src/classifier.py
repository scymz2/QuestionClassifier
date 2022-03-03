import torch.nn as nn

class Classifier(nn.Module):
    def __init_(self, n_input, n_hidden, n_output):
        super().__init_()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = self.predict(out2)

        return out
