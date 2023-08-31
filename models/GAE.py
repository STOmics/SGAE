import gc

import torch

torch.set_num_threads(20)
from torch import nn
from torch.nn import Module, Parameter
from torch_geometric.nn import GATConv

class GNNLayer(Module):

    def __init__(self, in_features, out_features, model_type='GCN'):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model_type = model_type
        # if args.name == "dblp":
        #     self.act = nn.Tanh()
        #     self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # else:
        if model_type == 'GCN':
            self.act = nn.Tanh()
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
            torch.nn.init.xavier_uniform_(self.weight)
        elif model_type == 'GAT':
            self.model = GATConv(in_features, out_features)

    # def forward(self, features, adj, active=False):
    #     if active:
    #         # if args.name == "dblp":
    #         #     support = self.act(F.linear(features, self.weight).cuda())  # add bias
    #         # else:
    #         # print('features.shape: ',features.shape)
    #         # print('self.weight.shape: ',self.weight.shape)
    #         support = self.act(torch.mm(features, self.weight))
    #     else:
    #         # if args.name == "dblp":
    #         #     support = F.linear(features, self.weight).cuda()  # add bias
    #         # else:
    #         support = torch.mm(features, self.weight)
    #     # torch.spmm: (前sparse × 后dense) or (前dense × 后dense)
    #     # torch.sparse.mm() a是稀疏矩阵，b是稀疏矩阵或者密集矩阵
    #     # print('type(adj): ', type(adj), 'type(support): ', type(support))
    #     output = torch.spmm(adj, support)
    #     return output
    #     # az = torch.spmm(adj, output)
    #     # return output, az
    def forward(self, features, adj, active=False):
        if self.model_type == 'GCN':
            if active:
                features = torch.mm(features, self.weight)
                features = self.act(features)
            else:
                features = torch.mm(features, self.weight)
            print('***', adj.shape, features.shape)
            output = torch.spmm(adj, features)
        elif self.model_type == 'GAT':
            if adj.layout == torch.strided:
                adj = torch.Tensor.to_sparse(adj)
            elif adj.layout == torch.sparse_csr or adj.layout == torch.sparse_coo:
                adj = adj
            output = self.model(features, adj.coalesce().indices(), adj.coalesce().values())
        return output


# IGAE encoder from DFCN
class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, n_inputs):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_inputs, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, gae_n_enc_3)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        # z_1 = self.gnn_1(x, adj, active=True)
        # z_2 = self.gnn_2(z_1, adj, active=True)
        # z = self.gnn_1(x, adj, active=True)
        # z = self.gnn_2(z, adj, active=True)
        # z = self.gnn_3(z, adj, active=False)
        # adj_z = self.s(torch.mm(z, z.t()))
        x = self.gnn_1(x, adj, active=True)
        x = self.gnn_2(x, adj, active=True)
        x = self.gnn_3(x, adj, active=False)
        # print('torch.cuda.current_device(): ',torch.cuda.current_device())
        # print('torch.cuda.memory_summary: ', torch.cuda.memory_summary(torch.cuda.current_device()))
        adj = self.s(torch.mm(x, x.t()))
        return x, adj


# IGAE decoder from DFCN
class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_inputs):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_5 = GNNLayer(gae_n_dec_2, gae_n_dec_3)
        self.gnn_6 = GNNLayer(gae_n_dec_3, n_inputs)
        self.s = nn.Sigmoid()
    def forward(self, z_igae, adj):
        # # z_1 = self.gnn_4(z_igae, adj, active=True)
        # # z_2 = self.gnn_5(z_1, adj, active=True)
        # z_hat = self.gnn_4(z_igae, adj, active=True)
        # z_hat = self.gnn_5(z_hat, adj, active=True)
        # z_hat = self.gnn_6(z_hat, adj, active=True)
        # z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        # return z_hat, z_hat_adj
        z_igae = self.gnn_4(z_igae, adj, active=True)
        z_igae = self.gnn_5(z_igae, adj, active=True)
        z_igae = self.gnn_6(z_igae, adj, active=True)
        adj = torch.mm(z_igae, z_igae.t())
        adj = self.s(adj)
        return z_igae, adj


class IGAE(nn.Module):

    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_inputs):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(gae_n_enc_1=gae_n_enc_1, gae_n_enc_2=gae_n_enc_2, gae_n_enc_3=gae_n_enc_3,
                                    n_inputs=n_inputs)

        self.decoder = IGAE_decoder(gae_n_dec_1=gae_n_dec_1, gae_n_dec_2=gae_n_dec_2, gae_n_dec_3=gae_n_dec_3,
                                    n_inputs=n_inputs)

    # def forward(self, x, adj):
    #     z_igae, z_igae_adj = self.encoder(x, adj)
    #     z_hat, z_hat_adj = self.decoder(z_igae, adj)
    #     adj_hat = z_igae_adj + z_hat_adj
    #     return z_hat, adj_hat, z_igae
    def forward(self, x, adj):
        z_igae, adj = self.encoder(x, adj)
        x, z_hat_adj = self.decoder(z_igae, adj)
        adj = adj + z_hat_adj
        return x, adj, z_igae