import os
import sys
sys.path.append(
    os.path.abspath("/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE"))

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from .AE import AE
from .GAE import IGAE
from .readout import Readout
from .opt import args


#
class SGAE(nn.Module):
    def __init__(self, n_node=None):
        super(SGAE, self).__init__()

        # Auto Encoder
        self.ae = AE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2, ae_n_enc_3=args.ae_n_enc_3,
            ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2, ae_n_dec_3=args.ae_n_dec_3, n_inputs=args.n_inputs,
            n_z=args.n_z)

        # Improved Graph Auto Encoder 
        self.gae = IGAE(gae_n_enc_1=args.gae_n_enc_1, gae_n_enc_2=args.gae_n_enc_2, gae_n_enc_3=args.gae_n_enc_3,
            gae_n_dec_1=args.gae_n_dec_1, gae_n_dec_2=args.gae_n_dec_2, gae_n_dec_3=args.gae_n_dec_3,
            n_inputs=args.n_inputs)

        # fusion parameter 
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, args.n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, args.n_z), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(args.n_clusters, args.n_z), requires_grad=True)

        # readout function
        self.R = Readout(k=args.n_clusters)

    # calculate the soft assignment distribution Q
    def q_distribute(self, Z, Z_ae, Z_igae):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]


    def q_distribute_5param(self, Z, Z_ae1, Z_ae2, Z_igae1, Z_igae2):
        """
        calculate the soft assignment distribution based on the embedding and the cluster centers
        Args:
            Z: fusion node embedding
            Z_ae: node embedding encoded by AE
            Z_igae: node embedding encoded by IGAE
        Returns:
            the soft assignment distribution Q
        """
        Z_ae = (Z_ae1 + Z_ae2) / 2
        Z_igae = (Z_igae1 + Z_igae2) / 2
        q = 1.0 / (1.0 + torch.sum(torch.pow(Z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(Z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(Z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]

    def forward(self, X_tilde1, Am, X_tilde2, Ad):

        # node embedding encoded by IGAE
        Z_igae1, A_igae1 = self.gae.encoder(X_tilde1, Am)
        Z_igae2, A_igae2 = self.gae.encoder(X_tilde2, Ad)

        # node embedding encoded by AE
        # Z_ae1 = self.ae.encoder(X_tilde1)
        # Z_ae2 = self.ae.encoder(X_tilde2)
        X_tilde1 = self.ae.encoder(X_tilde1)
        X_tilde2 = self.ae.encoder(X_tilde2)

        # cluster-level embedding calculated by readout function
        # Z_tilde_ae1 = self.R(Z_ae1)
        # Z_tilde_ae2 = self.R(Z_ae2)
        Z_tilde_igae1 = self.R(Z_igae1)
        Z_tilde_igae2 = self.R(Z_igae2)
        Z_tilde_ae1 = self.R(X_tilde1)
        Z_tilde_ae2 = self.R(X_tilde2)

        # linear combination of view 1 and view 2
        # Z_ae = (X_tilde1 + X_tilde2) / 2
        # Z_igae = (Z_igae1 + Z_igae2) / 2

        # node embedding fusion from DFCN
        # Z_i = self.a * Z_ae + self.b * Z_igae
        # Z_l = torch.spmm(Am, Z_i)
        # Z_l = self.a * Z_ae + self.b * Z_igae
        # Z_l = torch.spmm(Am, Z_l)
        Z_l = self.a * (X_tilde1 + X_tilde2) / 2 + self.b * (Z_igae1 + Z_igae2) / 2
        Z_l = torch.spmm(Am, Z_l)
        # S = torch.mm(Z_l, Z_l.t())
        # S = F.softmax(S, dim=1)
        # # Z_g = torch.mm(S, Z_l)
        # # Z = self.alpha * Z_g + Z_l
        # Z = torch.mm(S, Z_l)
        # Z = self.alpha * Z + Z_l
        Z = torch.mm(Z_l, Z_l.t())
        Z = F.softmax(Z, dim=1)
        # Z_g = torch.mm(S, Z_l)
        # Z = self.alpha * Z_g + Z_l
        Z = torch.mm(Z, Z_l)
        Z = self.alpha * Z + Z_l

        # AE decoding
        X_hat = self.ae.decoder(Z)

        # IGAE decoding
        # Z_hat, Z_adj_hat = self.gae.decoder(Z, Am)
        # sim = (A_igae1 + A_igae2) / 2
        # A_hat = sim + Z_adj_hat
        Z_hat, A_hat = self.gae.decoder(Z, Am)
        sim = (A_igae1 + A_igae2) / 2
        A_hat = sim + A_hat

        # node embedding and cluster-level embedding
        # Z_ae_all = [Z_ae1, Z_ae2, Z_tilde_ae1, Z_tilde_ae2]
        Z_ae_all = [X_tilde1, X_tilde2, Z_tilde_ae1, Z_tilde_ae2]
        Z_gae_all = [Z_igae1, Z_igae2, Z_tilde_igae1, Z_tilde_igae2]

        # the soft assignment distribution Q
        # Q = self.q_distribute(Z, Z_ae, Z_igae)
        # Q = self.q_distribute(Z, (X_tilde1 + X_tilde2) / 2, (Z_igae1 + Z_igae2) / 2)
        Q = self.q_distribute_5param(Z, X_tilde1, X_tilde2, Z_igae1, Z_igae2)

        # propagated embedding AZ_all and embedding Z_all
        # AZ_en = []
        # Z_en = []
        # for i in range(len(AZ_1)):
        #     AZ_en.append((AZ_1[i]+AZ_2[i])/2)
        #     Z_en.append((Z_1[i]+Z_2[i])/2)
        # AZ_all = [AZ_en, AZ_de]
        # Z_all = [Z_en, Z_de]

        # return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all
        return X_hat, Z_hat, A_hat, sim, Z_ae_all, Z_gae_all, Q, Z

