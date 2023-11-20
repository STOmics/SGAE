import gc
import os
import time

from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
from scanpy import read_h5ad
import torch
from .SGAE.utils.config import setup, set_path_models, print_setting
from .SGAE.utils.opt import args
from .SGAE.utils.utils import tensor_2_sparsetensor, preprocess, graph_construction_cluster, graph_construction_spatial_knn


if __name__ == '__main__':

    adata = read_h5ad(args.data_file)
    print(args.name)
    setup(args)

    adata, cm = preprocess(adata, args)
    adata.write_h5ad(osp.join(args.res_dir, 'preprocess_adata.h5ad'))

    print_setting(args)

    set_path_models(args, args.name)
    # Graph
    if os.path.exists(args.res_dir + "graphdict.pt"):
        graph_dict = torch.load(args.res_dir + "graphdict.pt")
    else:
        n_node = adata.shape[0]
        from .SGAE.utils.utils import preprocess_graph
        cms = ['louvain', 'leiden', 'kmeans']
        adj = graph_construction_cluster(adata.obs[cms[cm]].values.codes.reshape((n_node, 1)))
        adj += graph_construction_spatial_knn(adata.obsm['spatial'], k_nn=args.k_nn)

        graph_dict = preprocess_graph(adj, n_node, True, False)
        graph_dict['adj_norm'] = tensor_2_sparsetensor(graph_dict['adj_norm'])
        graph_dict['adj_ad'] = tensor_2_sparsetensor(graph_dict['adj_ad'])
        graph_dict['adj_org'] = tensor_2_sparsetensor(graph_dict['adj_org'])
        torch.save(graph_dict, args.res_dir + "graphdict.pt")

        # DataLoader
        from .SGAE.utils.utils import LoadDataset
        dataset = LoadDataset(adata.obsm['X_pca'])
        data_X = torch.FloatTensor(dataset.x.copy())  # .cuda()
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

        args.modelname = 'SGAE_Nosym'
        args.ae_modelname = 'ae'
        args.gae_modelname = 'gae_Nosym'
        args.pre_modelname = 'pre_Nosym'
        set_path_models(args, args.name)

        # train
        args.train_who = [1, 1, 1, 1]
        from .SGAE.trainers.train_HR import train_all_highRes
        train_all_highRes(train_loader=train_loader, data=data_X, graph_dict=graph_dict, args=args, adata=adata)
        gc.collect()
        torch.cuda.empty_cache()
