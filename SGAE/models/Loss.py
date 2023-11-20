import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

crossentropyloss = nn.CrossEntropyLoss()


# the reconstruction function: mse, mse, bce
def reconstruction_loss_BCE(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{rec}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """
    # loss_ae = F.binary_cross_entropy_with_logits(torch.sigmoid(X_hat), X)
    loss_ae = F.mse_loss(X_hat, X)

    # loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    # loss_w = F.binary_cross_entropy_with_logits(torch.sigmoid(Z_hat), torch.spmm(A_norm, X))
    loss_w = F.mse_loss(Z_hat, X)

    # loss_a = F.mse_loss(A_hat, A_norm)
    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()
    loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm)
    # when adjs ars sparse
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat.to_dense()), A_norm)
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm.to_dense())
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat.to_dense()), A_norm.to_dense())
    # loss_igae = loss_w + loss_a
    # loss_rec = loss_ae + loss_igae
    loss_rec = loss_ae + loss_w + loss_a
    return loss_rec


def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


def distribution_loss(Q, P):
    """
    calculate the clustering guidance loss L_{KL}
    Args:
        Q: the soft assignment distribution
        P: the target distribution
    Returns: L_{KL}
    """
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_correlation(Z_v1, Z_v2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    Z_v1 = F.normalize(Z_v1, dim=1)
    Z_v2 = F.normalize(Z_v2, dim=1)
    Z_v2 = Z_v2.t()
    # S = torch.mm(Z_v1, Z_v2)
    # return S
    gc.collect()
    torch.cuda.empty_cache()
    Z_v1 = torch.mm(Z_v1, Z_v2)
    return Z_v1


def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    loss = torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()
    return loss


def dicr_loss_2(Z_ae, Z_igae):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        # AZ: the propagated fusion embedding AZ
        # Z: the fusion embedding Z
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)

    # cross-view sample correlation matrix
    L_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    L_N_igae = cross_correlation(Z_igae[0], Z_igae[1])

    # loss of SCR
    # noinspection PyTypeChecker
    L_N_ae = correlation_reduction_loss(L_N_ae)
    L_N_igae = correlation_reduction_loss(L_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    L_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    L_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    L_F_ae = correlation_reduction_loss(L_F_ae)
    L_F_igae = correlation_reduction_loss(L_F_igae)

    loss_dicr = 0.1 * L_N_ae + 5 * L_N_igae + L_F_ae + L_F_igae

    return loss_dicr


def gcn_loss_high_res(data, adj, z_hat, adj_hat):
    z_hat = torch.sigmoid(z_hat)
    loss_w = F.binary_cross_entropy_with_logits(z_hat, data)
    adj_hat = torch.sigmoid(adj_hat)
    if adj.is_sparse:
        adj = adj.to_dense()
    loss_a = F.binary_cross_entropy_with_logits(adj_hat, adj)
    loss = loss_w + loss_a
    return loss


def sgae_loss_bce(X, adj_norm, X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, Q, lambda_value):
    L_DICR = dicr_loss_2(Z_ae_all, Z_gae_all)
    L_REC = reconstruction_loss_BCE(X, adj_norm, X_hat, Z_hat, A_hat)
    L_KL = distribution_loss(Q, target_distribution(Q[0].data))
    loss = L_DICR + L_REC + lambda_value * L_KL
    return loss


def pretrain_loss_bce(data, adj, x_hat, z_hat, adj_hat, z_ae=None, z_igae=None):
    loss_1 = F.binary_cross_entropy_with_logits(torch.sigmoid(x_hat), data)
    loss_2 = F.binary_cross_entropy_with_logits(torch.sigmoid(z_hat), data)
    if adj.is_sparse:
        adj = adj.to_dense()
    loss_3 = F.binary_cross_entropy_with_logits(torch.sigmoid(adj_hat), adj)
    loss = loss_1 + loss_2 + loss_3
    return loss
