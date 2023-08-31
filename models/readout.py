import torch
import torch.nn as nn


# readout function
class Readout(nn.Module):
    def __init__(self, k):
        super(Readout, self).__init__()
        self.k = k

    def forward(self, Z):
        # calculate cluster-level embedding
        Z_tilde = []

        # step1: split the nodes into K groups
        # step2: average the node embedding in each group
        n_node = Z.shape[0]
        step = n_node // self.k
        for i in range(0, n_node, step):
            if n_node - i < 2 * step:
                Z_tilde.append(torch.mean(Z[i:n_node], dim=0))
                break
            else:
                Z_tilde.append(torch.mean(Z[i:i + step], dim=0))

        # the cluster-level embedding
        Z_tilde = torch.cat(Z_tilde, dim=0)
        return Z_tilde.view(1, -1)

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    @staticmethod
    def forward(seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

