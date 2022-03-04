import torch.nn as nn


class Classifier(nn.Module):
    def __init_(self, n_input, n_hidden, n_output):
        super().__init_()
        self.net = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return self.net(x)
