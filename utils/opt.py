import argparse

parser = argparse.ArgumentParser(description='SGAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="dblp")
parser.add_argument('--gpu_id', type=str, default='3')
parser.add_argument('--modelname', type=str, default="SGAE")
# parser.add_argument('--project_dir', type=str,
#                     default="/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE/")
parser.add_argument('--writer', type=str, default="")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--transport_rate', type=float, default=0.2)

# hyperparameters
parser.add_argument('--n_clusters', type=int, default=20)
parser.add_argument('--k_nn', type=int, default=1)
# alpha_value: alpha value for graph diffusion
parser.add_argument('--alpha_value', type=float, default=0.2)

# training parameters
parser.add_argument('--train_who', type=int, default=[1, 1, 1, 1], nargs='+')  #
parser.add_argument('--epoch', type=int, default=1000) # 500
parser.add_argument('--all_epochs', type=int, default=[1000, 1000, 1000, 1000], nargs='+')
parser.add_argument('--patience', type=int, default=30) #

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr_p', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-4)

# loss weights
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=int, default=0.01)
parser.add_argument('--omega', type=float, default=0.1)
parser.add_argument('--lambda_value', type=float, default=10)
parser.add_argument('--gamma_value', type=float, default=1e3)
parser.add_argument('--with_topoloss', type=list, default=[False, False, False, False])
parser.add_argument('--with_sparse_regular', type=list, default=[False, False, False, False])

# model parameters
parser.add_argument('--n_inputs', type=int, default=100)
parser.add_argument('--n_z', type=int, default=20)

# AE structure parameter\
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)
parser.add_argument('--shuffle', type=bool, default=True)

# IGAE structure parameter\
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
# parser.add_argument('--fmi', type=float, default=0)
# parser.add_argument('--nmi', type=float, default=0)
# parser.add_argument('--ari', type=float, default=0)
# parser.add_argument('--acc', type=float, default=0)
# parser.add_argument('--f1', type=float, default=0)
args = parser.parse_args()
