import os
import random
import numpy as np
import torch


def setup(args):
    """
    setup
    - name: the name of dataset
    - device: CPU / GPU
    - seed: random seed
    - n_clusters: num of cluster
    - n_inputs: dimension of feature
    - alpha_value: alpha value for graph diffusion
    - lambda_value: lambda value for clustering guidance
    - gamma_value: gamma value for propagation regularization
    - lr: learning rate
    Return: None
    """
    print("data_name: ", args.name, "setting:")
    setup_seed(args.seed)

    ## protocal

    # test data capacity

    if 'Data_Cap' in args.name:
        args.n_clusters = 20
        args.n_inputs = 50
        args.n_z = 20

    # SLIDE-seq
    elif 'slideseqv2_MH' in args.name:
        args.n_clusters = 7
        args.n_inputs = 50
        args.n_z = 30
    elif 'slideseqv2_MOB' in args.name:
        args.n_clusters = 20
        args.n_inputs = 50
        args.n_z = 20
    elif 'slideseq_' in args.name:
        args.n_clusters = 9
        # 糖尿病1型小鼠
        if 'mouse_diabet_1' in args.name or 'MD1' in args.name: args.n_clusters = 9
    elif args.name == '10x':
        args.data_path = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/vgae/generated_data/'
        args.data_name = 'V1_Breast_Cancer_Block_A_Section_1'
        args.data_file = args.data_path + args.data_name + '/'
        args.n_clusters = 20
    elif args.name == 'DLPFC':
        args.n_clusters = 8
        args.lr = 1e-5
    elif args.name in ['DLPFC_151507', 'DLPFC_151508', 'DLPFC_151509', 'DLPFC_151510', 'DLPFC_151673', 'DLPFC_151674',
                       'DLPFC_151675', 'DLPFC_151676']:
        args.n_clusters = 7
        args.n_inputs = 200
        # args.lambda_I = 1
        args.n_z = 50  # best
        args.eval_graph_n = 20
        args.save_root = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE/DLPFC/'
    elif args.name in ['DLPFC_151669', 'DLPFC_151670', 'DLPFC_151671', 'DLPFC_151672']:
        args.n_clusters = 5
        args.n_inputs = 200
        args.n_z = 50
        args.eval_graph_n = 20
        args.save_root = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE/DLPFC/'
    elif args.name == 'MERFISH':
        args.n_clusters = 5
    elif 'MERFISH_CTL' in args.name or 'MERFISH_contex_lt' in args.name or 'MERFISH_CLT_topo' in args.name:
        args.n_clusters = 22
        args.n_inputs = 50
    elif 'MERFISH_contex' in args.name:
        args.n_clusters = 23
    elif 'seqFISH' in args.name:
        args.n_clusters = 22
        args.n_inputs = 50
        args.n_z = 20
    elif 'ST' in args.name:
        args.n_clusters = 10

    # Stereo-seq
    elif 'ME' in args.name:
        args.n_clusters = 12
    elif 'Drosophila' in args.name:
        args.n_clusters = 10
    elif 'MB_high_capture' in args.name or 'MB_hc' in args.name:
        args.n_clusters = 33  # 小鼠脑
    elif 'MB_fn' in args.name:
        args.n_clusters = 33  # 小鼠脑
    elif 'mouse_embryo' in args.name:
        args.n_clusters = 12
    elif 'ME_E165' in args.name:
        args.n_clusters = 25
    elif 'brain_8times' in args.name:
        args.n_clusters = 7
    elif 'MB_3D' in args.name:
        args.n_clusters = 33
    elif args.name in ['SS200000141TL_B5', 'SS200000128TR_E2', 'SS200000141TL_A4'] or 'MB_E2' in args.name:
        args.n_clusters = 33
        args.n_inputs = 50
        # args.n_z = 20
    elif 'human_' in args.name:
        args.n_clusters = 10
    elif args.name == 'MBW':
        args.n_clusters = 8  # (128852, 37345)
    elif 'E14' in args.name:
        args.n_clusters = 10
    elif 'Drosophila_3D' in args.name:
        args.n_clusters = 10
        args.n_inputs = 50
        args.n_z = 20

    # Clinical
    elif 'RE_' in args.name:
        if '10' in args.name:
            args.n_clusters = 12
        else:
            args.n_clusters = 7
    elif 'black' in args.name or 'Black' in args.name:
        args.n_clusters = 10
    elif 'LC' in args.name:
        args.n_clusters = 20
    elif 'pancancer' in args.name or 'Pancancer' in args.name:
        args.n_clusters = 17
    elif 'PDAC' in args.name:
        args.n_clusters = 20  # 32
    elif 'BRCA' in args.name:
        args.n_clusters = 20
    elif 'Liver' in args.name:
        args.n_clusters = 10

    # Plant
    elif 'MZ' in args.name:
        args.n_clusters = 10
    elif 'zhongzi' in args.name or 'ziyepei' in args.name:
        args.n_clusters = 9
    elif args.name == 'r3scRNA':
        args.n_clusters = 4
    elif args.name == 'whole_brain':
        args.n_clusters = 33
    elif args.name == 'DLPFC_3d':
        args.n_clusters = 7
    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("dataset       : ")
        print("device        : ")
        print("random seed   : ")
        print("clusters      : ")
        print("alpha value   : ")
        print("lambda value  : ")
        print("gamma value   : ")
        print("learning rate : ")
        print("------------------------------")
        # exit(0)

    # from torch.utils.tensorboard import SummaryWriter
    # from datetime import datetime
    # args.writer = SummaryWriter(args.res_dir + 'logs/' + args.name + '_' + args.modelname + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '_log')
    # un_limit()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.ae_modelname = 'ae'
    args.gae_modelname = 'gae'
    args.pre_modelname = 'pretrain'
    args.modelname = 'SGAE'

    args.project_dir = os.getcwd()
    # args.res_dir =  mk_dir(args.project_dir + 'res/' + args.name + '/')
    # if 'nn' in args.name:
    #     args.res_dir = mk_dir(args.project_dir + 'res/' + args.name[:-4] + '/')
    #     args.ae_model_save_path = args.res_dir + args.name + '_' + args.ae_modelname + '.pkl'
    #     args.gae_model_save_path = args.res_dir + args.name + '_' + args.gae_modelname + '.pkl'
    #     args.pre_model_save_path = args.res_dir + args.name + '_' + args.pre_modelname + '.pkl'
    #     args.model_save_path = args.res_dir + args.name + '_' + args.modelname + '.pkl'
    # else:
    # if not os.path.exists(args.project_dir + 'res/' + args.name + '/'):

    args.res_dir = mk_dir(os.path.join(args.project_dir, 'res', args.name))
    # else: args.res_dir = args.project_dir + 'res/' + args.name + '/'
    args.preds_dir = args.res_dir
    args.figs_dir = args.res_dir
    args.embs_dir = args.res_dir
    args.models_dir = args.res_dir
    set_path_models(args, args.name)

    if args.n_z != args.gae_n_enc_3:
        args.gae_n_enc_3 = args.n_z
        args.gae_n_dec_1 = args.n_z



def print_setting(args):
    print("------------------------------")
    print("dataset       : {}".format(args.name))
    print("device        : {}".format(args.device))
    print("random seed   : {}".format(args.seed))
    print("clusters      : {}".format(args.n_clusters))
    print("k_nn          : {}".format(args.k_nn))
    print("epoch         : {}".format(args.epoch))
    print("n_inputs      : {}".format(args.n_inputs))
    print("n_z           : {}".format(args.n_z))
    print("alpha value   : {}".format(args.alpha_value))
    print("lambda value  : {}".format(args.lambda_value))
    print("gamma value   : {}".format(args.gamma_value))
    print("learning rate : {}".format(args.lr))
    print("result dir    : {}".format(args.res_dir))
    print("------------------------------")


def un_limit():
    from os import system
    cmd = 'ulimit -S -s unlimited'
    system(cmd)


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def mk_dir(input_path):
    if not os.path.isdir(input_path):
        os.makedirs(input_path)
    return input_path


def set_path_models(args, data_name):
    # args.res_dir = mk_dir(args.project_dir + 'res/' + data_name + '/')
    # args.ae_model_save_path = args.res_dir + data_name + '_' + args.ae_modelname + '.pkl'
    # args.gae_model_save_path = args.res_dir + data_name + '_' + args.gae_modelname + '.pkl'
    # args.pre_model_save_path = args.res_dir + data_name + '_' + args.pre_modelname + '.pkl'
    # args.model_save_path = args.res_dir + data_name + '_' + args.modelname + '.pkl'
    # args.preds_dir = args.res_dir
    # args.figs_dir = args.res_dir
    # args.embs_dir = args.res_dir
    # args.models_dir = args.res_dir

    # pretrain ae
    if '_' in data_name and 'nn' in data_name:
        args.ae_model_save_path = os.path.join(args.res_dir, data_name[:-4] + '_' + args.ae_modelname + '.pkl')
        args.ae_emb_save_path = os.path.join(args.res_dir, data_name[:-4] + '_' + args.ae_modelname + '_emb.pkl')
        args.ae_pred_save_path = os.path.join(args.res_dir, data_name[:-4] + '_' + args.ae_modelname + '_pred.txt')
        args.ae_res_save_path = os.path.join(args.res_dir, data_name[:-4] + '_' + args.ae_modelname + '_res.txt')
    else:
        args.ae_model_save_path = os.path.join(args.res_dir, data_name + '_' + args.ae_modelname + '.pkl')
        args.ae_emb_save_path = os.path.join(args.res_dir, data_name + '_' + args.ae_modelname + '_emb.pkl')
        args.ae_pred_save_path = os.path.join(args.res_dir, data_name + '_' + args.ae_modelname + '_pred.txt')
        args.ae_res_save_path = os.path.join(args.res_dir, data_name + '_' + args.ae_modelname + '_res.txt')

    # pretrain gae
    args.gae_model_save_path = os.path.join(args.res_dir, data_name + '_' + args.gae_modelname + '.pkl')
    args.gae_emb_save_path = os.path.join(args.res_dir, data_name + '_' + args.gae_modelname + '_emb.pkl')
    args.gae_pred_save_path = os.path.join(args.res_dir, data_name + '_' + args.gae_modelname + '_pred.txt')
    args.gae_res_save_path = os.path.join(args.res_dir, data_name + '_' + args.gae_modelname + '_res.txt')

    # pretrain
    args.pre_model_save_path = os.path.join(args.res_dir, data_name + '_' + args.pre_modelname + '.pkl')
    args.pre_emb_save_path = os.path.join(args.res_dir, data_name + '_' + args.pre_modelname + '_emb.pkl')
    args.pre_pred_save_path = os.path.join(args.res_dir, data_name + '_' + args.pre_modelname + '_pred.txt')
    args.pre_res_save_path = os.path.join(args.res_dir, data_name + '_' + args.pre_modelname + '_res.txt')

    # train
    args.model_save_path = os.path.join(args.res_dir, data_name + '_' + args.modelname + '.pkl')
    args.emb_save_path = os.path.join(args.res_dir, data_name + '_' + args.modelname + '_emb.pkl')
    args.pred_save_path = os.path.join(args.res_dir, data_name + '_' + args.modelname + '_pred.txt')
    args.res_save_path = os.path.join(args.res_dir, data_name + '_' + args.modelname + '_res.txt')
