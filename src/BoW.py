import torch.nn as nn


class BoW(nn.Module):
    def __init__(self, pretrain_or_not, freeze_or_not, embeddings, num_embeddings, embedding_dim):
        super().__init__()

        # define layers
        if pretrain_or_not:
            if freeze_or_not:
                self.ebdLayer = nn.EmbeddingBag.from_pretrained(embeddings=embeddings, freeze=True)
            else:
                self.ebdLayer = nn.EmbeddingBag.from_pretrained(embeddings, freeze=False)
        else:
            self.ebdLayer = nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        output = self.ebdLayer(x)
        return output
