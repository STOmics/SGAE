# import warnings
# warnings.filterwarnings('ignore')
import os
import sys
import time

sys.path.append(
    os.path.abspath("/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE"))
import tqdm
import joblib
import numpy as np
import gc

import torch
import torch.nn as nn
crossentropyloss = nn.CrossEntropyLoss()
import torch.nn.functional as F
from torch.optim import Adam

from models.Loss import pretrain_loss_bce, sgae_loss_bce, gcn_loss_high_res
from utils.utils import remove_edge, clustering_mob, plot_eval_clustering
from utils.gpu_memory_log import gpu_memory_log


def pretrain_ae_highRes(train_loader, args):
    ae_epochs = args.n_epochs
    from models.AE import AE
    model = AE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2, ae_n_enc_3=args.ae_n_enc_3,
               ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2, ae_n_dec_3=args.ae_n_dec_3,
               n_inputs=args.n_inputs, n_z=args.n_z)
    model = model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    args.chscore = 0
    args.sscore = 0
    args.dbscore = 100000
    args.inertia = 100000
    cnt = 0
    for epoch in tqdm.tqdm(range(ae_epochs)):
        for batch_idx, (x, _) in enumerate(train_loader):
            model.train()
            x = x.cuda()
            x_hat, z_hat = model(x)
            loss = 10 * F.mse_loss(x_hat, x)
            loss.backward()
            # evaluation
            model.eval()
            z_hat = z_hat.detach().cpu().numpy()
            with torch.no_grad():
                if epoch % 1 == 0:
                    chscore, sscore, dbscore, inertia, cluster_id = clustering_mob(x, z_hat, args.n_clusters)
                    if chscore > args.chscore or sscore > args.sscore or dbscore < args.dbscore or inertia < args.inertia:
                        args.chscore = max(chscore, args.chscore)
                        args.sscore = max(chscore, args.sscore)
                        args.dbscore = min(chscore, args.dbscore)
                        args.inertia = min(chscore, args.inertia)
                        torch.save(model.state_dict(), args.ae_model_save_path)
                        cnt = 0
                    else:
                        cnt += 1
                if cnt == args.patience:
                    print('early stopping!!!')
                    break
        optimizer.step()
        model.zero_grad(set_to_none=True)  # 模型参数梯度清零
        optimizer.zero_grad(set_to_none=True)  # 优化器参数梯度清零
    print("Finish optimization.")
    try:
        gpu_memory_log(gpu_log_file=args.name + "_" + args.ae_modelname + "gpu_mem.log",
                       device=torch.cuda.current_device())
    except ReferenceError:
        pass



def pretrain_gae_highRes(data, adj, args):
    """
    pretrain_gae_high_Resolution
    """
    gae_epochs = args.n_epochs
    from models.GAE import IGAE
    model = IGAE(gae_n_enc_1=args.gae_n_enc_1, gae_n_enc_2=args.gae_n_enc_2, gae_n_enc_3=args.gae_n_enc_3,
                 gae_n_dec_1=args.gae_n_dec_1, gae_n_dec_2=args.gae_n_dec_2, gae_n_dec_3=args.gae_n_dec_3,
                 n_inputs=args.n_inputs).cuda()
    args.chscore = 0
    args.sscore = 0
    args.dbscore = 100000
    args.inertia = 100000
    cnt = 0
    cluster_id1 = []
    optimizer = Adam(model.parameters(), lr=args.lr)
    data = data.cuda()
    adj = adj.cuda()
    for epoch in tqdm.tqdm(range(gae_epochs)):
        model.train()
        z_hat, adj_hat, z_igae = model(data, adj)
        loss = gcn_loss_high_res(data, adj, z_hat, adj_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # 优化器参数梯度清零
        model.zero_grad(set_to_none=True)  # 模型参数梯度清零
        # evaluation
        model.eval()
        with torch.no_grad():
            z_igae = z_igae.detach().cpu().numpy()
            if epoch % 1 == 0:
                chscore, sscore, dbscore, inertia, cluster_id = clustering_mob(data, z_igae, args.n_clusters)
                if chscore > args.chscore or sscore > args.sscore or dbscore < args.dbscore or inertia < args.inertia:
                    args.chscore = max(chscore, args.chscore)
                    args.sscore = max(chscore, args.sscore)
                    args.dbscore = min(chscore, args.dbscore)
                    args.inertia = min(chscore, args.inertia)
                    torch.save(model.state_dict(), args.gae_model_save_path)
                    joblib.dump(z_igae, args.gae_emb_save_path)
                    cluster_id1 = cluster_id
                    cnt = 0
                else:
                    cnt += 1
            if cnt == args.patience:
                print('early stopping!!!')
                break
    if len(cluster_id1) == 0:
        torch.save(model.state_dict(), args.gae_model_save_path)
        joblib.dump(z_igae, args.gae_emb_save_path)
    else:
        cluster_id = cluster_id1
    np.savetxt(args.gae_pred_save_path, cluster_id, fmt='%d', delimiter='\t')
    print("Finish optimization.")
    try:
        gpu_memory_log(gpu_log_file=args.name + "_" + args.gae_modelname + "gpu_mem.log",
                   device=torch.cuda.current_device())
    except ReferenceError:
        pass
    return cluster_id


def pretrain_highRes(data, adj, args):
    pre_epochs = args.n_epochs
    from models.Pre_model import Pre_model
    model = Pre_model(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2, ae_n_enc_3=args.ae_n_enc_3,
                      ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2, ae_n_dec_3=args.ae_n_dec_3,
                      gae_n_enc_1=args.gae_n_enc_1, gae_n_enc_2=args.gae_n_enc_2, gae_n_enc_3=args.gae_n_enc_3,
                      gae_n_dec_1=args.gae_n_dec_1, gae_n_dec_2=args.gae_n_dec_2, gae_n_dec_3=args.gae_n_dec_3,
                      n_inputs=args.n_inputs, n_z=args.n_z, n_clusters=args.n_clusters, n_node=data.size()[0])
    # tra = tra.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    args.chscore = 0
    args.sscore = 0
    args.dbscore = 100000
    args.inertia = 100000
    cnt = 0
    cluster_id1 = []
    data = data.cuda()
    adj = adj.cuda()
    model = model.cuda()
    for epoch in tqdm.tqdm(range(pre_epochs)):
        model.train()
        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data, adj)
        loss = pretrain_loss_bce(data, adj, x_hat, z_hat, adj_hat, z_ae, z_igae)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)  # 模型参数梯度清零
        optimizer.zero_grad(set_to_none=True)  # 优化器参数梯度清零
        # evaluation
        model.eval()
        with torch.no_grad():
            z_tilde = z_tilde.detach().cpu().numpy()
            if epoch % 1 == 0:
                chscore, sscore, dbscore, inertia, cluster_id = clustering_mob(data, z_tilde, args.n_clusters)
                if chscore > args.chscore or sscore > args.sscore or dbscore < args.dbscore or inertia < args.inertia:
                    args.chscore = max(chscore, args.chscore)
                    args.sscore = max(chscore, args.sscore)
                    args.dbscore = min(chscore, args.dbscore)
                    args.inertia = min(chscore, args.inertia)
                    torch.save(model.state_dict(), args.pre_model_save_path)
                    joblib.dump(z_tilde, args.pre_emb_save_path)
                    cluster_id1 = cluster_id
                    cnt = 0
                else:
                    cnt += 1
            if cnt == args.patience:
                print('early stopping!!!')
                break
    if len(cluster_id1) == 0:
        torch.save(model.state_dict(), args.pre_model_save_path)
        joblib.dump(z_tilde, args.pre_emb_save_path)
    else:
        cluster_id = cluster_id1
    np.savetxt(args.pre_pred_save_path, cluster_id, fmt='%d', delimiter='\t')
    try:
        gpu_memory_log(gpu_log_file=args.name + "_" + args.pre_modelname + "gpu_mem.log",
                       device=torch.cuda.current_device())
    except ReferenceError:
        pass
    print("Finish optimization.")
    return cluster_id


def train_main_highRes(X, graph_dict, args):
    """
    train our model
    Args:
        modelname
        data_name
        model: Dual Correlation Reduction Network
        X: input feature matrix
        graph_dict: graph_dict['adj_org'], graph_dict['adj_norm'], graph_dict['adj_ad']: A: input origin adj, A_norm: normalized adj, Ad: graph diffusion
        labels: input label
    Returns: acc, nmi, ari, f1
    """
    sgae_epochs = args.n_epochs

    from utils.utils import model_init, gaussian_noised_feature
    from models.SGAE_model import SGAE
    model = SGAE(n_node=X.shape[0])
    print("Training…")
    model = model.cuda()
    X = X.cuda()
    graph_dict['adj_norm'] = graph_dict['adj_norm'].cuda()
    # graph_dict['adj_org'] = graph_dict['adj_org'].cuda()
    # calculate embedding similarity and cluster centers
    sim, centers = model_init(model, X, graph_dict['adj_norm'], args.n_clusters, args.pre_model_save_path, args.name)
    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers, device=torch.cuda.current_device())
    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    if graph_dict['adj_org'].is_cuda:
        graph_dict['adj_org'] = graph_dict['adj_org'].detach().cpu().to_dense()
    else:
        graph_dict['adj_org'] = graph_dict['adj_org'].to_dense()
    sim = sim.detach().cpu()
    # X = X.detach().cpu()
    Am = remove_edge(graph_dict['adj_org'], sim)
    Am = Am.to_sparse()
    Am = Am.cuda()
    del sim
    gc.collect()
    graph_dict['adj_ad'] = graph_dict['adj_ad'].cuda()
    cluster_id1 = []
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    optimizer = Adam(model.parameters(), lr=args.lr)
    args.chscore = 0
    args.sscore = 0
    args.dbscore = 100000
    args.inertia = 100000
    cnt = 0
    for epoch in tqdm.tqdm(range(sgae_epochs)):
        model.train()
        # from torch.cuda.amp import autocast
        # with autocast():
        # RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)
        # input & output
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z = model(X_tilde1, graph_dict['adj_ad'], X_tilde2, Am)
        loss = sgae_loss_bce(X, graph_dict['adj_norm'], X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, args.lambda_value)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)  # 模型参数梯度清零
        optimizer.zero_grad(set_to_none=True)  # 优化器参数梯度清零
        # evaluation
        model.eval()
        with torch.no_grad():
            Z = Z.detach().cpu().numpy()
            if epoch == 0:
                try:
                    gpu_memory_log(gpu_log_file=args.name + "_" + args.modelname + "gpu_mem.log",
                                   device=torch.cuda.current_device())
                except ReferenceError:
                    pass
            if epoch % 1 == 0:
                del X_tilde1, X_tilde2, X_hat, Z_hat, A_hat, Z_ae_all, Z_gae_all, Q
                chscore, sscore, dbscore, inertia, cluster_id = clustering_mob(X, Z, args.n_clusters)
                if chscore > args.chscore or sscore > args.sscore or dbscore < args.dbscore or inertia < args.inertia:
                    args.chscore = max(chscore, args.chscore)
                    args.sscore = max(chscore, args.sscore)
                    args.dbscore = min(chscore, args.dbscore)
                    args.inertia = min(chscore, args.inertia)
                    torch.save(model.state_dict(), args.model_save_path)
                    joblib.dump(Z, args.emb_save_path)
                    cluster_id1 = cluster_id
                    cnt = 0
                else:
                    cnt += 1
            if cnt == args.patience:
                print('early stopping!!!')
                break

    if len(cluster_id1) == 0:
        torch.save(model.state_dict(), args.model_save_path)
        joblib.dump(Z, args.emb_save_path)
    else:
        cluster_id = cluster_id1
    np.savetxt(args.pred_save_path, cluster_id, fmt='%d', delimiter='\t')
    print("Finish optimization.")

    return cluster_id


def train_all_highRes(train_loader, data, graph_dict, args, adata=None):
    time_stat = {}
    gc.collect()
    torch.cuda.empty_cache()
    if args.train_who[0] == 1:
        print('=================================== GAE ===================================')
        start = time.time()
        labels2 = pretrain_gae_highRes(data, graph_dict['adj_norm'], args)
        end = time.time()
        time_gae = end - start
        print('gae训练时间：', time_gae)
        plot_eval_clustering(adata, labels2, args.gae_modelname, args.gae_emb_save_path, args)
        end_plot = time.time()
        time_stat['train_gae'] = round(time_gae)
        time_stat['plot_gae'] = round(end_plot - end)
        gc.collect()
        torch.cuda.empty_cache()

    if args.train_who[1] == 1:
        print('=================================== AE ===================================')
        start = time.time()
        pretrain_ae_highRes(train_loader, args)
        end = time.time()
        time_ae = end - start
        print('ae训练时间：', time_ae)
        time_stat['train_ae'] = round(time_ae)
        gc.collect()
        torch.cuda.empty_cache()

    if args.train_who[2] == 1:
        print('=================================== Pre ===================================')
        start = time.time()
        labels2 = pretrain_highRes(data, graph_dict['adj_norm'], args)
        end = time.time()
        time_pre = end - start
        print('pre训练时间：', time_pre)
        print('pre训练完成时间：', time_gae+time_ae+time_pre)
        plot_eval_clustering(adata, labels2, args.pre_modelname, args.pre_emb_save_path, args)
        end_plot = time.time()
        time_stat['train_pre'] = round(time_pre)
        time_stat['plot_pre'] = round(end_plot - end)
        gc.collect()
        torch.cuda.empty_cache()

    if args.train_who[3] == 1:
        print('=================================== SGAE ===================================')
        start = time.time()
        labels2 = train_main_highRes(data, graph_dict, args)
        end = time.time()
        time_sgae = end - start
        print('sgae训练时间：', time_sgae)
        print('sgae训练完成时间：', time_gae+time_ae+time_pre+time_sgae)

        plot_eval_clustering(adata, labels2, args.modelname, args.emb_save_path, args)
        end_plot = time.time()
        time_stat['train_sgae'] = round(time_sgae)
        time_stat['plot_sgae'] = round(end_plot - end)
        gc.collect()
        torch.cuda.empty_cache()
    # np.save('time_statistics_individual_' + str(args.k_nn) + '.npy', time_stat)



