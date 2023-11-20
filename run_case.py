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


def run_case(dataset, adata):
    args.name = dataset
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


if __name__ == '__main__':


    dataset = args.dataset

    if dataset == 'merfish':
        adata = read_h5ad('mouse1sample1_left_top.h5ad')
        run_case(dataset, adata)

    elif dataset == 'dlpfc':
        sample_list = ['151673', '151674', '151675', '151676', '151507', '151508', '151509', '151510', '151669',
                       '151670', '151671', '151672']

        for sample in sample_list:
            adata = read_h5ad(sample + '.h5ad')
            run_case(sample, adata)

    elif dataset == 'seqfish':
        adata = read_h5ad('seqFISH_lohoff2021integration_lohoff2020highly_seqFISH_mouse_Gastrulation_data.h5ad')
        run_case(dataset, adata)


    elif dataset == 'slideseq':
        adata = read_h5ad('slideSeqV2_MOB.h5ad')
        run_case(dataset, adata)

    elif dataset == 'mousebrain':
        adata = read_h5ad('SS200000128TR_E2.h5ad')
        run_case(dataset, adata)

    elif dataset == 'drosophila_14_16':
        adata = read_h5ad('E14-16h_a_count_normal_stereoseq.h5ad')
        run_case(dataset, adata)

    elif dataset == 'drosophila_16_18':
        adata = read_h5ad('E16-18h_a_count_normal_stereoseq.h5ad')
        run_case(dataset, adata)

    elif dataset == 'drosophila_l1':
        adata = read_h5ad('L1_a_count_normal_stereoseq.h5ad')
        run_case(dataset, adata)

    else:
        print('Please specify the *.h5ad file then use run_sgae.py.')
