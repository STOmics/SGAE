import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
crossentropyloss = nn.CrossEntropyLoss()

# l1 regularization
def l1_regularization(model):
    l1_loss = []
    # l2_loss = []
    for name, parameter in model.named_parameters():
        # if type(module) is nn.BatchNorm2d:
        l1_loss.append(torch.abs(parameter).sum())  # l2_loss.append((parameter ** 2).sum() / 2.0)
    return sum(l1_loss)  # , sum(l2_loss)

# sparse regularization loss
def reg_loss(model):
    # layers_count = len(model.layers)
    loss_reg = 0
    for name, param in model.named_parameters():
        # if 'gae' in name or 'ae' in name:
        if 'weight' in name:
            # LASSO
            loss_reg += param.abs().sum()  # RIDGE  # loss_reg += self._gcn_parameters[i].norm()**2  #  # print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    # for i in range(layers_count):
    #     # LASSO
    #     loss_reg += self._gcn_parameters[i].abs().sum()
    #     # RIDGE
    #     # loss_reg += self._gcn_parameters[i].norm()**2
    return loss_reg

# the reconstruction function: mse, mse, mse
def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
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
    loss_ae = F.mse_loss(X_hat, X)
    X = torch.spmm(A_norm, X)
    loss_w = F.mse_loss(Z_hat, X)
    loss_a = F.mse_loss(A_hat, A_norm)
    # loss_igae = loss_w + 0.1 * loss_a
    # loss_rec = loss_ae + loss_igae
    loss_rec = loss_ae + loss_w + 0.1 * loss_a
    return loss_rec

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
def reconstruction_loss_mask(X, A_norm, X_hat, Z_hat, A_hat,dropout_mask_data,dropout_mask_adj):
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
    # loss_ae = F.mse_loss(X_hat, X)

    loss_ae = torch.sum((X - X_hat).pow(2) * dropout_mask_data) / torch.sum(dropout_mask_data)
    # loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
    # loss_w = F.binary_cross_entropy_with_logits(torch.sigmoid(Z_hat), torch.spmm(A_norm, X))
    # loss_w = F.mse_loss(Z_hat, X)
    loss_w = torch.sum((X - Z_hat).pow(2) * dropout_mask_data) / torch.sum(dropout_mask_data)
    # loss_a = F.mse_loss(A_hat, A_norm)
    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()
    # loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm)

    loss_a = torch.sum((A_norm - A_hat).pow(2) * dropout_mask_adj) / torch.sum(dropout_mask_adj)
    # when adjs ars sparse
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat.to_dense()), A_norm)
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm.to_dense())
    #     loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat.to_dense()), A_norm.to_dense())
    # loss_igae = loss_w + loss_a
    # loss_rec = loss_ae + loss_igae
    loss_rec = loss_ae + loss_w + loss_a
    return loss_rec

def reconstruction_loss_DLPFC(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """

    loss_ae = F.mse_loss(X_hat, X)
    loss_w = F.mse_loss(Z_hat, X)
    # loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    # loss_a = gcn_loss(torch.sigmoid(A_hat, A_norm.to_dense()))
    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()
    loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm)
    loss_igae = loss_w + loss_a
    loss_rec = loss_ae + loss_igae
    return loss_rec

# (X, graph_dict["adj_norm"], X_hat, A_hat)


def reconstruction_coss_KLD(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def reconstruction_loss_entropy(X, A_norm, X_hat, Z_hat, A_hat):
    """
    reconstruction loss L_{}
    Args:
        X: the origin feature matrix
        A_norm: the normalized adj
        X_hat: the reconstructed X
        Z_hat: the reconstructed Z
        A_hat: the reconstructed A
    Returns: the reconstruction loss
    """
    # loss_ae = F.mse_loss(X_hat, X)
    X_hat = torch.sigmoid(X_hat)
    loss_ae = F.binary_cross_entropy_with_logits(X_hat, X)

    # loss_w = F.mse_loss(Z_hat, X)
    Z_hat = torch.sigmoid(Z_hat)
    loss_w = F.binary_cross_entropy_with_logits(Z_hat, X)

    # loss_a = F.mse_loss(A_hat, A_norm.to_dense())
    # loss_a = gcn_loss(torch.sigmoid(A_hat, A_norm.to_dense()))
    # loss_a = F.binary_cross_entropy_with_logits(torch.sigmoid(A_hat), A_norm.to_dense())
    A_hat = torch.sigmoid(A_hat)
    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()
    loss_a = F.binary_cross_entropy_with_logits(A_hat, A_norm)

    # loss_igae = loss_w + loss_a
    # loss_rec = loss_ae + loss_igae
    loss_rec =loss_ae + loss_w + loss_a
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

def r_loss(AZ, Z):
    """
    the loss of propagated regularization (L_R)
    Args:
        AZ: the propagated embedding
        Z: embedding
    Returns: L_R
    """
    loss = 0
    for i in range(2):
        for j in range(3):
            p_output = F.softmax(AZ[i][j], dim=1)
            q_output = F.softmax(Z[i][j], dim=1)
            log_mean_output = ((p_output + q_output) / 2).log()
            loss += (F.kl_div(log_mean_output, p_output, reduction='batchmean') + F.kl_div(log_mean_output, p_output,
                                                                                           reduction='batchmean')) / 2
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
    # S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    # S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    L_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    L_N_igae = cross_correlation(Z_igae[0], Z_igae[1])

    # loss of SCR
    # noinspection PyTypeChecker
    # L_N_ae = correlation_reduction_loss(S_N_ae)
    # L_N_igae = correlation_reduction_loss(S_N_igae)
    L_N_ae = correlation_reduction_loss(L_N_ae)
    L_N_igae = correlation_reduction_loss(L_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    # S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    # S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())
    L_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    L_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    # L_F_ae = correlation_reduction_loss(S_F_ae)
    # L_F_igae = correlation_reduction_loss(S_F_igae)
    L_F_ae = correlation_reduction_loss(L_F_ae)
    L_F_igae = correlation_reduction_loss(L_F_igae)

    # if args.name == "dblp" or args.name == "acm":
    # L_N = 0.01 * L_N_ae + 10 * L_N_igae
    # L_F = 0.5 * L_F_ae + 0.5 * L_F_igae
    # else:
    # L_N = 0.1 * L_N_ae + 5 * L_N_igae
    # L_F = L_F_ae + L_F_igae

    # propagated regularization
    # L_R = r_loss(AZ, Z)

    # loss of DICR
    # loss_dicr = L_N + L_F + args.gamma_value * L_R
    # print('L_N:',L_N.item(),'L_F:',L_F.item(),'L_R:',L_R.item())
    # loss_dicr = L_N + L_F
    loss_dicr = 0.1 * L_N_ae + 5 * L_N_igae + L_F_ae + L_F_igae

    return loss_dicr

def dicr_loss_DLPFC(Z_ae, Z_igae, AZ, Z):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    # loss of SCR
    # noinspection PyTypeChecker
    L_N_ae = correlation_reduction_loss(S_N_ae)
    # noinspection PyTypeChecker
    L_N_igae = correlation_reduction_loss(S_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    # noinspection PyTypeChecker
    L_F_ae = correlation_reduction_loss(S_F_ae)
    # noinspection PyTypeChecker
    L_F_igae = correlation_reduction_loss(S_F_igae)

    # if args.name == "dblp" or args.name == "acm":
    #     L_N = 0.01 * L_N_ae + 10 * L_N_igae
    #     L_F = 0.5 * L_F_ae + 0.5 * L_F_igae
    # else:
    L_N = 0.1 * L_N_ae + 5 * L_N_igae
    L_F = L_F_ae + L_F_igae

    # propagated regularization
    # L_R = r_loss(AZ, Z)

    # loss of DICR
    # loss_dicr = L_N + L_F + args.gamma_value * L_R
    # print('L_N:',L_N.item(),'L_F:',L_F.item(),'L_R:',L_R.item())
    loss_dicr = L_N + L_F

    return loss_dicr

def dicr_loss(Z_ae, Z_igae, AZ, Z, gamma_value):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        AZ: the propagated fusion embedding AZ
        Z: the fusion embedding Z
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    S_N_ae = cross_correlation(Z_ae[0], Z_ae[1])
    S_N_igae = cross_correlation(Z_igae[0], Z_igae[1])
    # loss of SCR
    # noinspection PyTypeChecker
    L_N_ae = correlation_reduction_loss(S_N_ae)
    L_N_igae = correlation_reduction_loss(S_N_igae)

    # Feature-level Correlation Reduction (FCR)
    # cross-view feature correlation matrix
    S_F_ae = cross_correlation(Z_ae[2].t(), Z_ae[3].t())
    S_F_igae = cross_correlation(Z_igae[2].t(), Z_igae[3].t())

    # loss of FCR
    # noinspection PyTypeChecker
    L_F_ae = correlation_reduction_loss(S_F_ae)
    L_F_igae = correlation_reduction_loss(S_F_igae)
    # if args.name == "dblp" or args.name == "acm":
    #     L_N = 0.01 * L_N_ae + 10 * L_N_igae
    #     L_F = 0.5 * L_F_ae + 0.5 * L_F_igae
    # else:
    #     L_N = 0.1 * L_N_ae + 5 * L_N_igae
    #     L_F = L_F_ae + L_F_igae

    # propagated regularization
    # if 'DLPFC' in args.name or 'RE' in args.name:
    #     L_R = 0
    #     L_N = 0.1 * L_N_ae + 5 * L_N_igae
    #     L_F = L_F_ae + L_F_igae
    # else:
    L_R = r_loss(AZ, Z)

    # loss of DICR
    # L_F = L_F_ae + L_F_igae
    # L_N = L_N_ae + L_N_igae
    # loss_dicr = L_N + L_F
    # print('L_N:', L_N.item(), 'L_F:', L_F.item(), 'L_R:', L_R.item())
    L_N = 0.01 * L_N_ae + 10 * L_N_igae
    L_F = 0.5 * L_F_ae + 0.5 * L_F_igae
    loss_dicr = L_N + L_F + gamma_value * L_R
    return loss_dicr

def gcn_loss_high_res_noAhat(data, z_hat):
    # loss_w = F.mse_loss(z_hat, data)
    # # print('type(adj_hat),type(adj):')
    # # print(type(adj_hat),type(adj))
    # loss_a = F.mse_loss(adj_hat, adj)
    # z_hat = torch.sigmoid(z_hat)
    loss = F.binary_cross_entropy_with_logits(z_hat, data)
    return loss

def gcn_loss_high_res(data, adj, z_hat, adj_hat):
    # loss_w = F.mse_loss(z_hat, data)
    # # print('type(adj_hat),type(adj):')
    # # print(type(adj_hat),type(adj))
    # loss_a = F.mse_loss(adj_hat, adj)
    z_hat = torch.sigmoid(z_hat)
    loss_w = F.binary_cross_entropy_with_logits(z_hat, data)
    adj_hat = torch.sigmoid(adj_hat)
    # print(type(adj))
    if adj.is_sparse:
        adj = adj.to_dense()
    loss_a = F.binary_cross_entropy_with_logits(adj_hat, adj)
    # print('Loss_w {:.2f}'.format(loss_w.item()), ' Loss_a {:.2f}'.format(loss_a.item()))
    loss = loss_w + loss_a
    return loss

def gcn_loss_high_res_mask(data, adj, z_hat, adj_hat,dropout_mask_data,dropout_mask_adj):

    loss_w = torch.sum((data - z_hat).pow(2) * dropout_mask_data) / torch.sum(dropout_mask_data)
    if adj.is_sparse:
        adj = adj.to_dense()

    loss_a = torch.sum((adj - adj_hat).pow(2) * dropout_mask_adj) / torch.sum(dropout_mask_adj)
    loss = loss_w + loss_a
    return loss

def sgae_loss_high_res(X, adj_norm, X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, Q, gamma_value, lambda_value):
    # L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, gamma_value)
    L_DICR = dicr_loss_2(Z_ae_all, Z_gae_all)
    L_REC = reconstruction_loss(X, adj_norm, X_hat, Z_hat, A_hat)
    L_KL = distribution_loss(Q, target_distribution(Q[0].data))
    loss = L_DICR + L_REC + lambda_value * L_KL
    print('L_DICR {:.2f}'.format(L_DICR.item()), ' L_REC {:.2f}'.format(L_REC.item()), ' L_KL {:.2f}'.format(L_KL.item()))
    return loss

def sgae_loss_bce(X, adj_norm, X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, Q, gamma_value, lambda_value):
    # L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, gamma_value)
    L_DICR = dicr_loss_2(Z_ae_all, Z_gae_all)
    L_REC = reconstruction_loss_BCE(X, adj_norm, X_hat, Z_hat, A_hat)
    L_KL = distribution_loss(Q, target_distribution(Q[0].data))
    loss = L_DICR + L_REC + lambda_value * L_KL
    return loss
def sgae_loss_mask(X, adj_norm, X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, Q, gamma_value, lambda_value,dropout_mask_data, dropout_mask_adj):
    # L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, gamma_value)
    L_DICR = dicr_loss_2(Z_ae_all, Z_gae_all)
    L_REC = reconstruction_loss_mask(X, adj_norm, X_hat, Z_hat, A_hat,dropout_mask_data, dropout_mask_adj)
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
def pretrain_loss_mask(data, adj, x_hat, z_hat, adj_hat, dropout_mask_data, dropout_mask_adj):
    # dropout_mask = (data != 0).to(torch.float32)
    # if not dropout_mask.is_cuda:
    #     dropout_mask = dropout_mask.cuda()
    loss_1 = torch.sum((data - x_hat).pow(2) * dropout_mask_data) / torch.sum(dropout_mask_data)
    loss_2 = torch.sum((data - z_hat).pow(2) * dropout_mask_data) / torch.sum(dropout_mask_data)
    # dropout_mask = (adj != 0).to(torch.float32)
    # if not dropout_mask.is_cuda:
    #     dropout_mask = dropout_mask.cuda()
    # loss_3 = torch.sum((adj - adj_hat).pow(2) * dropout_mask_adj) / torch.sum(dropout_mask_adj)
    loss_3 = torch.sum((adj_hat-adj).pow(2) * dropout_mask_adj) / torch.sum(dropout_mask_adj)
    loss = loss_1 + loss_2 + loss_3
    return loss

def pretrain_loss_mse(data, adj, x_hat, z_hat, adj_hat, z_ae, z_igae, args):
    loss_1 = F.mse_loss(x_hat, data)
    loss_2 = F.mse_loss(z_hat, data)
    loss_3 = F.mse_loss(adj_hat, adj)
    loss_4 = F.mse_loss(z_ae, z_igae)
    loss_5 = F.mse_loss(z_hat, torch.spmm(adj, data))
    # loss_1 = F.binary_cross_entropy_with_logits(torch.sigmoid(x_hat), data)
    # loss_2 = F.binary_cross_entropy_with_logits(torch.sigmoid(z_hat), data)
    # loss_3 = F.binary_cross_entropy_with_logits(torch.sigmoid(adj_hat), adj)
    # if with_topoloss:
    loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3 + args.omega * loss_4 + args.beta * loss_5
    return loss
