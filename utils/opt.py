import argparse

parser = argparse.ArgumentParser(description='SGAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
parser.add_argument('--name', type=str, default="dblp")
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--modelname', type=str, default="SGAE")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--transport_rate', type=float, default=0.2)

# hyperparameters
parser.add_argument('--n_clusters', type=int, default=20)
parser.add_argument('--k_nn', type=int, default=3)
# alpha_value: alpha value for graph diffusion
parser.add_argument('--alpha_value', type=float, default=0.2)

# training parameters
parser.add_argument('--train_who', type=int, default=[1, 1, 1, 1], nargs='+')  #
parser.add_argument('--n_epochs', type=int, default=1000, nargs='+')
parser.add_argument('--patience', type=int, default=30) #

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)

# loss weights
parser.add_argument('--lambda_value', type=float, default=10)
parser.add_argument('--gamma_value', type=float, default=1e3)

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

args = parser.parse_args()
