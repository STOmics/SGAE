import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


from .opt import args
from .AE import AE
from .GAE import IGAE


class Pre_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, gae_n_enc_1, gae_n_enc_2,
            gae_n_enc_3, gae_n_dec_1, gae_n_dec_2, gae_n_dec_3, n_inputs, n_z, n_clusters, n_node=None):
        super(Pre_model, self).__init__()

        self.ae = AE(ae_n_enc_1=ae_n_enc_1, ae_n_enc_2=ae_n_enc_2, ae_n_enc_3=ae_n_enc_3, ae_n_dec_1=ae_n_dec_1,
                     ae_n_dec_2=ae_n_dec_2, ae_n_dec_3=ae_n_dec_3, n_inputs=n_inputs, n_z=n_z)

        self.ae.load_state_dict(torch.load(args.ae_model_save_path, map_location='cuda:{}'.format(torch.cuda.current_device())))
        self.ae = self.ae.eval()

        self.gae = IGAE(gae_n_enc_1=gae_n_enc_1, gae_n_enc_2=gae_n_enc_2, gae_n_enc_3=gae_n_enc_3,
                        gae_n_dec_1=gae_n_dec_1, gae_n_dec_2=gae_n_dec_2, gae_n_dec_3=gae_n_dec_3, n_inputs=n_inputs)
        self.gae.load_state_dict(torch.load(args.gae_model_save_path, map_location='cuda:{}'.format(torch.cuda.current_device())))
        self.gae = self.gae.eval()
        # if 'MB_fn' in args.name:
        #     pretrained_dict = self.gae.load_state_dict(torch.load(args.gae_model_save_path, map_location='cuda:{}'.format(torch.cuda.current_device())))
        #     model_dict = self.gae.state_dict()
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     pretrained_dict = {k: v.T for k, v in pretrained_dict.items()}
        #     model_dict.update(pretrained_dict)
        #     self.gae.load_state_dict(model_dict)

        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        # z_i = self.a * z_ae + self.b * z_igae
        # z_l = torch.spmm(adj, z_i)
        z_l = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_l)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)

        z_tilde = torch.mm(s, z_l)
        z_tilde = self.gamma * z_tilde + z_l
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj
        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde
