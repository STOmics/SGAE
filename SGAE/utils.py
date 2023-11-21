# import warnings
# warnings.filterwarnings('ignore')
# import matplotlib
# matplotlib.use('Agg')
# plt.rcParams['savefig.dpi'] = 800
import gc
import os.path as osp
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.core.dtypes.common import is_categorical_dtype
# import scanpy as sc
from scanpy import pl, pp, tl, read_h5ad, read_visium, external
from scipy import sparse
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, calinski_harabasz_score, silhouette_score, \
    davies_bouldin_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn import preprocessing


# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from opt import args
# sc.set_figure_params(scanpy=True, dpi=80, dpi_save=100, frameon=False)
#
# def mk_dir(input_path):
#     if not osp.exists(input_path):
#         makedirs(input_path)
#     return input_path

def convert_annotation(annotation):
    le = preprocessing.LabelEncoder()
    le.fit(annotation)
    annotation = le.transform(annotation)
    return annotation


def get_adata(data_file):
    print('data_file:', data_file)

    if '.h5ad' in data_file:
        adata = read_h5ad(data_file)
    elif 'slideSeqV2_MOB' in data_file:
        # adata = get_slideseq_v2_adata(data_file)
        adata = get_slideseq_v2_nofilter_adata(data_file)
    elif 'slideseq' in data_file:
        adata = get_slideseq_adata(data_file)
    # elif 'fn' in data_file:
    #     adata = get_fn_adata(data_file)
    elif '.gem.gz' in data_file:
        adata = get_fn_adata(data_file)
    elif '3D' in data_file:
        adata = get_Drosophila_adata(data_file)
    else:
        adata = None
    return adata

def get_Drosophila_adata(data_file):
    adata_list = []
    files = find_files_in_curDIR_givenType_givenKeyword(data_file, 'h5ad', file_type='*.h5ad')
    for i in range(len(files)):
        adata = read_h5ad(files[i])
        adata_list.append(adata)
    return adata_list

def get_BRCA_adata(data_file, args):
    import scanpy as sc
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    # print(adata)
    '''
    AnnData object with n_obs × n_vars = 201809 × 40196
        obs: 'orig.ident', 'x', 'y'
        obsm: 'spatial'
    counts.shape:  (201809, 40196)
    idx.shape:  (201809,)
    '''
    # preprocess

    adata = preprocess(adata, args)
    gc.collect()
    return adata


def get_ME_Times_adata(periods, data_name, data_path):
    # data
    periods = ['E9.5_E1S1', 'E10.5_E1S1', 'E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1', 'E15.5_E1S1',
               'E16.5_E1S1']
    # periods = ['E9.5_E1S1', 'E10.5_E1S1', 'E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1']
    times = len(periods)
    data_name = 'ME_' + str(times) + 'times'
    res_dir = osp.join('/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/SGAE',
                       'res', data_name)
    data_path = osp.join(res_dir, 'ME_8Times_adata_concatenated_harmonypy.h5ad')
    if osp.exists('ME_8Times_adata_concatenated_harmonypy.h5ad'):
        adata = read_h5ad(data_path)

    else:
        data_path = '/jdfssz2/ST_BIOINTEL/P20Z10200N0157/liuke/STOmicsDB/study/rawdata/10.Mosta/Embryo_data'
        print('------ concatenate ------')
        for period in range(len(periods)):
            filename = periods[period] + '.MOSTA.h5ad'
            data_file = osp.join('%s/%s' % (data_path, filename))
            temp_adata = read_h5ad(data_file)
            n_cell = temp_adata.shape[0]
            temp_adata.obs['period'] = pd.Categorical([periods[period] for _ in range(n_cell)])
            batch_period = pd.Categorical([period for _ in range(n_cell)])
            temp_adata.obs['batch_period'] = batch_period
            tl.louvain(temp_adata)
            if period == 0:
                adata = temp_adata
                adata.var_names_make_unique()
                adata.obs_names_make_unique()
            else:
                # x
                # noinspection PyUnboundLocalVariable
                if adata.obsm['spatial'].shape[1] == 2:
                    adata.obs['x'] = adata.obsm['spatial'][:, 0]
                    adata.obs['y'] = adata.obsm['spatial'][:, 1]
                    temp_adata.obs['x'] = temp_adata.obsm['spatial'][:, 0]
                    temp_adata.obs['y'] = temp_adata.obsm['spatial'][:, 1]
                else:
                    adata.obs['x'] = adata.obsm['spatial'][0, :]
                    adata.obs['y'] = adata.obsm['spatial'][1, :]
                    temp_adata.obs['x'] = temp_adata.obsm['spatial'][0, :]
                    temp_adata.obs['y'] = temp_adata.obsm['spatial'][1, :]
                max_x_pr = adata.obs['x'].max()  #
                min_x_cu = temp_adata.obs['x'].min()  #
                delta_x = max_x_pr - min_x_cu + 10
                temp_adata.obs['x'] += delta_x
                # y
                mean_y_pr = adata.obs['y'].mean()  #
                mean_y_cu = temp_adata.obs['y'].mean()  #
                delta_y = mean_y_pr - mean_y_cu
                temp_adata.obs['y'] += delta_y
                # spatial
                temp_adata.obsm['spatial'] = np.vstack((temp_adata.obs['x'], temp_adata.obs['y'])).T
                # concatenate
                adata = adata.concatenate(temp_adata)
                adata.var_names_make_unique()
                adata.obs_names_make_unique()
                adata.obs.fillna(0)
        # noinspection PyUnboundLocalVariable
        print(adata)
        print(adata.shape)
        # (520815, 23761)
        adata = adata.copy()
        adata.obs['batch_period'] = adata.obs['batch_period'].astype('category')
        adata.obs['louvain'] = adata.obs['louvain'].astype('category')
        # noinspection PyTypeChecker
        adata.write_h5ad(osp.join(res_dir, 'ME_8Times_adata_concatenated.h5ad'))

        # print('------ plot ------')
        # import matplotlib
        # matplotlib.use('Agg')
        # # plot_adata_multi(adata, res_dir)
        # pl.spatial(adata, color='annotation', spot_size=1, img=None, img_key=None, show=False)
        # save_path = osp.join(res_dir, data_name + '_annotation1.pdf')
        # plt.savefig(save_path, bbox_inches='tight')
        # plt.close()

        print('------ annotation ------')
        adata.obs['annotation'] = adata.obs['annotation'].astype('category')
        annotation = np.array(adata.obs['annotation'])
        cell_type = np.unique(annotation)
        n_clusters = len(cell_type)
        # 44

        print('------ preprocess ------')
        pp.recipe_zheng17(adata)

        print('------ pca ------')
        tl.pca(adata)

        print('------ batch effect ------')
        # 进行批次处理
        # pip install harmonypy
        external.pp.harmony_integrate(adata, key='batch_period')

        #
        pp.neighbors(adata)
        adata.obs['louvain_all'] = adata.obs['louvain_all'].astype('category')
        tl.umap(adata)
        pl.umap(adata, color=['annotation', 'batch_period', 'louvain', 'louvain_all'], legend_fontsize=8,
                save=osp.join(res_dir, 'annotation_batch_period_louvain.png'))

        # noinspection PyTypeChecker
        adata.write_h5ad(osp.join(res_dir, 'ME_8Times_adata_concatenated_harmonypy.h5ad'))
    return adata

def get_ME_T_adata_all(data_file, timepoint_list, timepoint_list_no, args):
    import scanpy as sc
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    '''
        AnnData object with n_obs × n_vars = 520815 × 23761
        obs: 'annotation', 'timepoint', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes'
        var: 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts'
        uns: 'annotation_colors'
        obsm: 'spatial'
        layers: 'count'
        '''
    # print('adata.shape: ',adata.shape)
    # preprocess
    adata = preprocess(adata, args)
    if len(timepoint_list_no) != 0:
        for time in timepoint_list_no:
            adata = adata[adata.obs['timepoint'] != time]
    print('adata.shape: ', adata.shape)

    # labels
    labels = np.array(adata.obs['annotation'].values)
    cell_types = list(np.unique(labels))
    labels = [cell_types.index(i) for i in labels]
    adata.obsm['annotation_num'] = np.array(labels).astype('int')
    return adata


def get_ME_T_adata(data_file, args):
    import scanpy as sc
    # timepoint_list = ['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5']  # ,'E15.5','E16.5'
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    '''
        AnnData object with n_obs × n_vars = 520815 × 23761
        obs: 'annotation', 'timepoint', 'n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes'
        var: 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts'
        uns: 'annotation_colors'
        obsm: 'spatial'
        layers: 'count'
        '''
    print('adata.shape: ', adata.shape)
    # preprocess
    adata = preprocess(adata, args)
    # node_nums = adata.shape[0]
    # adata = adata[:node_nums-2351175]
    adata = adata[adata.obs['timepoint'] != 'E15.5']
    adata = adata[adata.obs['timepoint'] != 'E16.5']
    print('adata.shape: ', adata.shape)

    # labels
    labels = np.array(adata.obs['annotation'].values)
    cell_types = list(np.unique(labels))
    labels = [cell_types.index(i) for i in labels]
    adata.obsm['annotation_num'] = np.array(labels)
    # # dataframe
    # data_df = pd.DataFrame(adata.X,index=list(range(adata.shape[0])), columns = list(range(adata.shape[1])))
    # data_df['coor_x'] = adata.obsm['spatial'][:, 0]
    # data_df['coor_y'] = adata.obsm['spatial'][:, 1]
    # # data_df['coor_z'] = coor_z
    # data_df['labels'] = labels
    # print('data_df.shape: ', data_df.shape)
    # data_df.dropna()
    # print('data_df.shape: ', data_df.shape)
    # # print(data_df)
    # # print(data_df.iloc[100001:])
    # # exit(0)
    # # del adata
    return adata


def get_merfish_adata(data_file, args):
    import scanpy as sc
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()

    # coor
    coor_x = adata.obs['X'].values
    coor_y = adata.obs['Y'].values
    adata.obsm['spatial'] = np.vstack((coor_x, coor_y)).T

    # annotation
    labels = adata.obs['subclass'].values.codes
    cell_types = list(set(labels))
    args.n_clusters = len(cell_types)
    labels = labels.astype('int')
    adata.obsm['annotation_num'] = labels
    return adata


def get_fn_adata(data_path):
    # data_path =
    from gem2h5ad import gem2h5ad
    adata = gem2h5ad(data_path)
    adata.var_names_make_unique()
    return adata


# def get_LC_data(data_path):
#
#     # df = pd.read_csv(data_path,compression='gzip', comment='#') .gem.gz
#     df = pd.read_csv(data_path, comment='#', delimiter='\t')
#     print(df)
#     # print(df.columns)
#     '''
#     geneID      x      y  MIDCount  label  tag
#     '''
#     # df = df[df['tag']!=0]
#     # df = df[df['label']!=0]
#     # df = df[df['label']==0]
#     # df.query('A=="foo"')
#     # df.query('A=="foo" | A=="bar"')
#     # df = df.query('label==0') # 背景 x shape (27,523,190, 46913)
#     # df = df.query('tag==0') # X.shape:  (33980510, 47636)
#     # df = df.query('label!=0 and tag !=0')
#     # df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')
#     # label_type = list(np.unique(df['label']))
#     # print(len(label_type)) # 106,347
#     # for j in label_type:
#     # df = df.query('label==j')
#     # df = df[['x_y', 'geneID', 'MIDCount']]
#     df = df.query('tag!=0')
#     df.dropna(inplace=True)
#     df.rename(columns={'label': 'cell_id'}, inplace=True)
#     cell_list = df["label"].astype('category')
#     gene_list = df["geneID"].astype('category')
#     data = df["MIDCount"].to_numpy()
#     row = cell_list.cat.codes.to_numpy()
#     col = gene_list.cat.codes.to_numpy()
#     obs = pd.DataFrame(index=cell_list.cat.categories)
#     var = pd.DataFrame(index=gene_list.cat.categories)
#     X = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var)))
#     print('X.shape: ', X.shape)
#
#     x_cells = []
#     y_cells = []
#     for ob in cell_list.cat.categories:
#         # print('CellID: ',ob)
#         x_aver = np.average(df[df['label'] == ob]['x'].to_numpy())
#         y_aver = np.average(df[df['label'] == ob]['y'].to_numpy())
#         x_cells.append(x_aver)
#         y_cells.append(y_aver)
#     # gdf = None
#     # if data.bin_type == 'cell_bins':
#     #     df.rename(columns={'label': 'cell_id'}, inplace=True)
#     #     gdf = parse_cell_bin_coor(df)
#     # else:
#     #     df = parse_bin_coor(df, bin_size)
#     cells = df['cell_id'].unique()
#     genes = df['geneID'].unique()
#     print('len of x_cells: ', len(cells))
#     print('len of y_cells: ', len(genes))
#     # idx = range(X.shape[0])
#     print('len of count_cells: ', len(idx))
#     coor = np.array([x_cells, y_cells]).reshape((len(idx), 2))
#     labels = [1 for _ in range(idx)]
#
#     plot_labels('LC', coor, labels, 'jpg')
#     return X, coor
def get_LC_adata(data_path, args):
    import scanpy as sc
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    # labels
    adata = preprocess(adata, args)
    # labels = [1 for _ in range(len(adata))]
    return adata


def get_MB_adata(dir_path, section, args):
    import scanpy as sc
    print('section: ', section)
    data_path = dir_path + section

    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()

    # # preprocess
    # adata = preprocess(adata, args)

    # labels
    labels = np.array(adata.obsm['annotation_au']).squeeze()
    cell_types = list(np.unique(labels))
    args.clusters = len(labels)
    labels2 = np.array([cell_types.index(j) for j in labels]).astype('int')
    adata.obsm['annotation_num'] = labels2
    return adata


def get_MB(dir_path, section, args):
    import scanpy as sc

    print('section: ', section)
    data_path = dir_path + section

    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()

    # labels
    labels = np.array(adata.obsm['annotation_au']).squeeze()
    cell_types = list(np.unique(labels))
    args.clusters = len(labels)
    labels2 = np.array([cell_types.index(j) for j in labels]).astype('int')
    adata.obsm['annotation_num'] = labels2
    return adata


def get_MB_3D_data_adata(dir_path, args):
    import scanpy as sc
    # sections = ['SS200000141TL_B5.h5ad', 'SS200000128TR_E2.h5ad', 'SS200000141TL_A4.h5ad']

    data_path = dir_path + 'SS200000141TL_B5.h5ad'
    adata_b5 = sc.read_h5ad(data_path)
    adata_b5.var_names_make_unique()
    adata_b5 = preprocess(adata_b5, args)
    adata_b5.obs['Batch'] = 'Batch1'

    data_path = dir_path + 'SS200000128TR_E2.h5ad'
    adata_e2 = sc.read_h5ad(data_path)
    adata_e2.var_names_make_unique()
    adata_e2 = preprocess(adata_e2, args)
    adata_e2.obs['Batch'] = 'Batch2'

    data_path = dir_path + 'SS200000141TL_A4.h5ad'
    adata_a4 = sc.read_h5ad(data_path)
    adata_a4.var_names_make_unique()
    adata_a4 = preprocess(adata_a4, args)
    adata_a4.obs['Batch'] = 'Batch3'

    adata = sc.AnnData.concatenate(adata_b5, adata_e2, adata_a4)
    labels = adata.obsm['annotation_au'].to_numpy()
    cell_types = list(np.unique(labels))
    if 'str' in str(type(cell_types[0])):
        labels = [cell_types.index(labels[j]) for j in range(len(labels))]
    adata.obsm['annotation_num'] = np.array(labels)

    return adata


def get_black_data(data_file, args):
    import scanpy as sc
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()

    # preprocess
    adata = preprocess(adata, args)

    idx = list(range(adata.X.shape[0]))

    # dataframe
    data_df = pd.DataFrame(adata.X, index=idx, columns=list(range(adata.X.shape[1])))
    data_df['coor_x'] = adata.obsm['spatial'][:, 0]
    data_df['coor_y'] = adata.obsm['spatial'][:, 1]

    return data_df


def get_black_adata(data_file):
    import scanpy as sc
    adata = sc.read_h5ad(data_file)
    # preprocess
    # adata = preprocess(adata,args)
    return adata


def get_slideseq_v2_nofilter_adata(file_fold):
    import scanpy as sc
    # data file
    # file_fold = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/slideSeqV2_MOB/'
    counts_file = file_fold + 'Puck_200127_15.digital_expression.txt'
    coor_file = file_fold + 'Puck_200127_15_bead_locations.csv'
    used_barcode_file = file_fold + 'used_barcodes.txt'
    # data
    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    coor_df = pd.read_csv(coor_file, index_col=3)
    coor_df.drop(coor_df.columns[coor_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
    adata.obsm["spatial"] = coor_df.to_numpy()
    # print('adata.shape:')
    # print(adata.shape)
    # used_barcode = pd.read_csv(used_barcode_file, sep='\t', header=None)
    # used_barcode = used_barcode[0]
    # noinspection PyTypeChecker
    # adata = adata[used_barcode,]
    # print('used_barcode shape:')
    # print(adata.shape)
    return adata


def get_slideseq_v2_adata(file_fold):
    import scanpy as sc
    # data file
    # file_fold = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/slideSeqV2_MOB/'
    counts_file = file_fold + 'Puck_200127_15.digital_expression.txt'
    coor_file = file_fold + 'Puck_200127_15_bead_locations.csv'
    used_barcode_file = file_fold + 'used_barcodes.txt'
    # data
    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    coor_df = pd.read_csv(coor_file, index_col=3)
    coor_df.drop(coor_df.columns[coor_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    adata = sc.AnnData(counts.T)
    adata.var_names_make_unique()
    coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
    adata.obsm["spatial"] = coor_df.to_numpy()
    # print('adata.shape:')
    # print(adata.shape)
    used_barcode = pd.read_csv(used_barcode_file, sep='\t', header=None)
    used_barcode = used_barcode[0]
    # noinspection PyTypeChecker
    adata = adata[used_barcode,]
    # print('used_barcode shape:')
    # print(adata.shape)
    return adata


def get_slideseq_adata(dir_path):
    import scanpy as sc
    count_file = dir_path + "MappedDGEForR_T4_Trimmed.csv"
    coor_file = dir_path + "BeadLocationsForR_T4_Trimmed.csv"
    anno_file = dir_path + "Puck_T4_bead_maxct_df.csv"

    # counts
    f = open(count_file)
    _ = f.readline()
    counts = []
    line = f.readline()
    while line:
        tmp = line.split(',')
        tmp[-1] = tmp[-1].replace('\n', '')
        # tmp = [int(j) for j in tmp[2:]]
        counts.append(tmp[2:])
        line = f.readline()
    f.close()
    counts = np.array(counts).astype(np.float32)
    # print('counts: ', counts.shape)

    # coor
    coor = np.loadtxt(coor_file, dtype='str', encoding='utf-8', delimiter=',', skiprows=1)[:, 1:]
    print('coor: ', coor.shape)
    # print(coor)
    coor = coor.astype(np.float32)

    # anno
    anno = np.loadtxt(anno_file, dtype='str', encoding='utf-8', delimiter=',', skiprows=1)[:, -1]
    anno = anno.astype('int')
    # print('anno: ', anno.shape)

    # Anndata
    adata = sc.AnnData(counts)
    adata.obsm['spatial'] = coor
    adata.obs['annotation'] = pd.Categorical(anno)
    adata.obsm['annotation_num'] = anno
    return adata


def get_slideseq_data(dir_path):
    count_file = dir_path + "MappedDGEForR_T4_Trimmed.csv"
    coor_file = dir_path + "BeadLocationsForR_T4_Trimmed.csv"
    anno_file = dir_path + "Puck_T4_bead_maxct_df.csv"

    # counts
    f = open(count_file)
    _ = f.readline()
    counts = []
    line = f.readline()
    while line:
        tmp = line.split(',')
        tmp[-1] = tmp[-1].replace('\n', '')
        # tmp = [int(j) for j in tmp[2:]]
        counts.append(tmp[2:])
        line = f.readline()
    f.close()
    counts = np.array(counts).astype(np.float32)
    # counts = StandardScaler().fit_transform(counts)
    print('counts: ', counts.shape)

    idx = range(counts.shape[0])
    # coor
    coor = np.loadtxt(coor_file, dtype='str', encoding='utf-8', delimiter=',', skiprows=1)[:, 1:]
    print('coor: ', coor.shape)
    # print(coor)

    # anno
    anno = np.loadtxt(anno_file, dtype='str', encoding='utf-8', delimiter=',', skiprows=1)[:, -1]
    print('anno: ', anno.shape)
    # print(anno)

    # dataframe
    data_df = pd.DataFrame(counts, index=idx, columns=range(counts.shape[1]))
    data_df['coor_x'] = coor[:, 0].astype(np.float32)
    data_df['coor_y'] = coor[:, 1].astype(np.float32)
    data_df['labels'] = anno.astype('int')
    return data_df


def get_pancancer_data(data_file):
    # data
    # cell bin数据读取根据cellID当索引，相同的cellID的x_y坐标取均值
    df = pd.read_table(data_file, skiprows=6)
    # df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')
    # remove x_ y: 0.0 0.0(remove background spots with big area and noisy spots with tiny area)
    # df = df[df['x_y'] != '0.0_0.0']
    df2 = df[['CellID', 'geneID', 'MIDCount']]
    cell_list = df2["CellID"].astype('category')
    gene_list = df2["geneID"].astype('category')
    data = df2["MIDCount"].to_numpy()
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    obs = pd.DataFrame(index=cell_list.cat.categories)
    var = pd.DataFrame(index=gene_list.cat.categories)
    counts = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var))).todense()
    print('counts.shape: ', counts.shape)
    # print('counts: ')
    # print(counts[:5])
    # coor
    x_cells = []
    y_cells = []
    for ob in cell_list.cat.categories:
        # print('CellID: ',ob)
        x_aver = np.average(df[df['CellID'] == ob]['x'].to_numpy())
        y_aver = np.average(df[df['CellID'] == ob]['y'].to_numpy())
        x_cells.append(x_aver)
        y_cells.append(y_aver)
    print('len of x_cells: ', len(x_cells))
    print('len of y_cells: ', len(y_cells))
    idx = range(counts.shape[0])

    # coor = np.array([x_cells, y_cells]).reshape((len(idx), 2))

    columns = range(counts.shape[1])
    data_df = pd.DataFrame(counts, index=idx, columns=columns)
    data_df['coor_x'] = x_cells
    data_df['coor_y'] = y_cells
    print(data_df)

    return data_df


def get_mz_data(data_file):
    df = pd.read_table(data_file)
    print('df.columns: ', df.columns)
    df['x_y'] = df['x'].astype('str') + '_' + df['y'].astype('str')
    # remove x_ y: 0.0 0.0(remove background spots with big area and noisy spots with tiny area)
    df = df[df['x_y'] != '0.0_0.0']
    df = df[['x_y', 'geneID', 'MIDCount']]
    cell_list = df["x_y"].astype('category')
    gene_list = df["geneID"].astype('category')
    data = df["MIDCount"].to_numpy()
    # print(data.shape)
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    obs = pd.DataFrame(index=cell_list.cat.categories)
    var = pd.DataFrame(index=gene_list.cat.categories)
    X = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var))).todense()
    print('X.shape: ', X.shape)

    # coor
    coor = list(cell_list.cat.categories)
    coor = [i.split('_') for i in coor]
    coor = np.array(coor)
    coor = coor.astype('float')

    data_df = pd.DataFrame(X, index=list(range(X.shape[0])), columns=list(range(X.shape[1])))
    data_df['coor_x'] = coor[:, 0]
    data_df['coor_y'] = coor[:, 1]

    return data_df


# @nb.jit(parallel=True)
def generate_xy(df, cell_list):
    x_cells = []
    y_cells = []
    for ob in cell_list:
        # print('CellID: ',ob)
        x_aver = np.average(df[df['CellID'] == ob]['x'].to_numpy())
        y_aver = np.average(df[df['CellID'] == ob]['y'].to_numpy())
        print(ob, x_aver, y_aver)
        x_cells.append(x_aver)
        y_cells.append(y_aver)
    print('len of x_cells: ', len(x_cells))
    print('len of y_cells: ', len(y_cells))
    return x_cells, y_cells


def get_pancancer_adata(data_file):
    import scanpy as sc
    # data
    # cell bin数据读取根据cellID当索引，相同的cellID的x_y坐标取均值
    df = pd.read_table(data_file, skiprows=6)
    df2 = df[['CellID', 'geneID', 'MIDCount']]
    cell_list = df2["CellID"].astype('category')
    gene_list = df2["geneID"].astype('category')
    data = df2["MIDCount"].to_numpy()
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    obs = pd.DataFrame(index=cell_list.cat.categories)
    var = pd.DataFrame(index=gene_list.cat.categories)
    counts = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var))).todense()
    data_df = pd.DataFrame(counts)
    data_df['coor_x'] = df.groupby('CellID').mean()['x']
    data_df['coor_y'] = df.groupby('CellID').mean()['y']
    # print(data_df)
    # print(data_df['coor_x'].shape)
    # print(data_df['coor_y'].shape)
    # coor = np.hstack((data_df['coor_x'].values, data_df['coor_y'].values))
    adata = sc.AnnData(counts)
    adata.obsm['spatial'] = data_df.iloc[:, -2:].values
    # print(adata)
    # print(adata.obsm['spatial'])
    adata = preprocess_2(adata)
    print(adata)
    return adata


def get_pancancer_data_numba(data_file):
    # data
    # cell bin数据读取根据cellID当索引，相同的cellID的x_y坐标取均值
    df = pd.read_table(data_file, skiprows=6)
    df2 = df[['CellID', 'geneID', 'MIDCount']]
    cell_list = df2["CellID"].astype('category')
    gene_list = df2["geneID"].astype('category')
    data = df2["MIDCount"].to_numpy()
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    obs = pd.DataFrame(index=cell_list.cat.categories)
    var = pd.DataFrame(index=gene_list.cat.categories)
    counts = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var))).todense()
    data_df = pd.DataFrame(counts)
    data_df['coor_x'] = df.groupby('CellID').mean()['x']
    data_df['coor_y'] = df.groupby('CellID').mean()['y']
    # print(data_df)
    return data_df





def get_data_DLPFC(args):
    import ot
    # import scanpy as sc
    input_dir = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/vgae/data/DLPFC/151676/'
    # path = args.sample_data_path
    # sample = args.sample
    # input_dir = path
    sample = '151676'

    # count_file = input_dir + sample + '_filtered_feature_bc_matrix.h5'
    path_truth = input_dir + sample + '_truth3.txt'
    # input_dir = r'D:\文件\代码\algorithm_GRN\data\DLPFC\151676'
    # path_truth = r'D:\文件\代码\algorithm_GRN\data\DLPFC\151676\151676_truth3.txt'
    # section_id = '151676'
    # count_file = '151676_filtered_feature_bc_matrix.h5'

    # read data
    adata = read_visium(path=input_dir, count_file=sample + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    # Normalization
    # pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    pp.filter_genes_dispersion(adata, flavor="seurat", n_top_genes=3000)
    pp.normalize_total(adata, target_sum=1e4)
    pp.log1p(adata)

    Ann_df = pd.read_csv(osp.join(input_dir, sample + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    truth = np.loadtxt(path_truth, dtype=str)[:, 1]
    truth = truth.astype('int')

    # adj
    coor = adata.obsm['spatial']
    dist_matrix = ot.dist(coor, metric='euclidean')
    adj = 1 - dist_matrix / np.max(dist_matrix)
    if args.flag != 1:
        adj = normalize_adj(adj, symmetry=False)
    else:
        pass
    # X
    X_data = adata_Vars.X.toarray()[:, ]
    return coor, adj, X_data, truth


def get_data_DLPFC_index_new_new(sample):
    import ot
    import scanpy as sc

    input_dir = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/DLPFC_all/' + sample + '/'
    adata_file = sample + '_filtered_feature_bc_matrix.h5'
    truth_file = input_dir + sample + '_truth_num.txt'

    # read data
    adata = sc.read_visium(path=input_dir, count_file=adata_file)
    adata.var_names_make_unique()
    # num_nodes = adata.shape[0]
    # n_genes = adata.shape[1]
    # print('cell number: ', num_nodes, 'gene number: ', n_genes)

    # Normalization
    # pp.higly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # pp.filter_genes_dispersion(adata, flavor="seurat", n_top_genes=3000)
    # pp.normalize_total(adata, target_sum=1e4)
    # pp.log1p(adata)
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    truth = np.loadtxt(truth_file, dtype=int)  # [:, 1]
    # truth = convert_annotation(truth, sample)
    # truth = truth.astype('int')

    # adj
    coor = adata.obsm['spatial']
    # metric delpfc 151676 gae chebyshev 0.14  canberra曼哈顿距离的加权 0.18 sqeuclidean 0.17 euclidean 0.16 cityblock 0.17 jaccard 0.17
    # dist_matrix = ot.dist(coor, metric='canberra')
    dist_matrix = ot.dist(coor, metric='sqeuclidean')
    # dist_matrix = sparse.csr_matrix(dist_matrix)
    # dist_matrix = torch.FloatTensor(dist_matrix).cuda()
    dist_matrix = numpy_to_torch(dist_matrix)
    adj = 1 - dist_matrix / torch.max(dist_matrix)
    X_data = adata.X.toarray()
    return coor, adj, X_data, truth


def get_data_DLPFC_index(sample):
    import ot
    import scanpy as sc
    input_dir = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/DLPFC_all/' + sample + '/'
    adata_file = sample + '_filtered_feature_bc_matrix.h5'
    truth_file = input_dir + sample + '_truth_num.txt'

    # read data
    adata = sc.read_visium(path=input_dir, count_file=adata_file)
    adata.var_names_make_unique()
    num_nodes = adata.shape[0]
    n_genes = adata.shape[1]
    print('cell number: ', num_nodes, 'gene number: ', n_genes)
    truth = np.loadtxt(truth_file, dtype=int)  # [:, 1]
    # truth = convert_annotation(truth, sample)
    # truth = truth.astype('int')

    # adj
    coor = adata.obsm['spatial']
    dist_matrix = ot.dist(coor, metric='euclidean')
    adj = 1 - dist_matrix / np.max(dist_matrix)

    # if args.flag != 1:
    #     adj = normalize_adj(adj, self_loop=True, symmetry=False)
    # else:
    #     pass
    # X
    X_data = adata.X.toarray()
    return coor, adj, X_data, truth


def get_data_DLPFC_2():
    import scanpy as sc
    input_dir = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/vgae/data/DLPFC/151676/'
    # path_truth = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/vgae/data/DLPFC/151676/151676_truth.txt'

    sample = '151676'

    # count_file = input_dir + sample + '_filtered_feature_bc_matrix.h5'
    path_truth = input_dir + sample + '_truth3.txt'
    # input_dir = r'D:\文件\代码\algorithm_GRN\data\DLPFC\151676'
    # path_truth = r'D:\文件\代码\algorithm_GRN\data\DLPFC\151676\151676_truth3.txt'
    # section_id = '151676'
    # count_file = '151676_filtered_feature_bc_matrix.h5'

    # read data
    adata = sc.read_visium(path=input_dir, count_file=sample + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    coor = adata.obsm['spatial']

    pp.filter_genes(adata, min_cells=3)
    adata_X = pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = pp.scale(adata_X)
    adata_X = pp.pca(adata_X, n_comps=100)
    adata_X.astype('float32')
    # Ann_df = pd.read_csv(osp.join(input_dir, sample + '_truth.txt'), sep='\t', header=None, index_col=0)
    # Ann_df.columns = ['Ground Truth']
    # adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    #
    # if 'highly_variable' in adata.var.columns:
    #     adata = adata[:, adata.var['highly_variable']]
    # else:
    #     adata = adata
    truth = np.loadtxt(path_truth, dtype=str)[:, 1]
    truth = truth.astype('int')
    # print(truth)

    # adj
    # dist_matrix = ot.dist(coor, metric='euclidean')
    # # dist2corr = 1 - dist_matrix / np.max(dist_matrix)
    # adj = 1 - dist_matrix / np.max(dist_matrix)
    adj_org, adj_norm_m1, graph_dict = graph_construction(coor)

    # X
    # X_data = adata_Vars.X.toarray()[:, ]
    # return adj_org, adj_norm_m1, X_data, truth
    # return adj_0, adj_0, X_data, cell_type_indeces
    return adj_org, adj_norm_m1, adata_X, truth


def get_data_MBW():
    import ot
    import scanpy as sc
    data_path = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/MB_w11.1.2/w11.1.2spatial.h5ad'
    coor_path = "/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/MB_w11.1.2/w11.1.2coor.csv"
    adata = sc.read_h5ad(data_path)
    X_data = adata.X
    coor = np.loadtxt(coor_path, delimiter='\t', dtype=np.int32)
    dist_matrix = ot.dist(coor)
    # dist_matrix = sparse.csr_matrix(dist_matrix)
    dist_matrix = numpy_to_torch(dist_matrix, sparse=True)
    adj = 1 - dist_matrix / torch.max(dist_matrix.to_dense())
    # n_clusters = 8
    return coor, adj, X_data


def load_ST_file(file_fold, proj_idx, load_images=True, file_Adj=None):
    import scanpy as sc
    count_file = proj_idx + '_filtered_feature_bc_matrix.h5'
    adata = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = osp.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_col_in_fullres',
                             'pxl_row_in_fullres', ]
        positions.index = positions['barcode']
        adata.obs = adata.obs.join(positions, how="left")
        adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata




def concat_adata(periods, data_path):
    print('------ concatenate ------')
    for period in range(len(periods)):
        filename = periods[period] + '.MOSTA.h5ad'
        data_file = osp.join('%s/%s' % (data_path, filename))
        temp_adata = read_h5ad(data_file)
        n_cell = temp_adata.shape[0]
        temp_adata.obs['period'] = pd.Categorical([periods[period] for _ in range(n_cell)])
        batch_period = pd.Categorical([period for _ in range(n_cell)])
        temp_adata.obs['batch_period'] = batch_period
        if period == 0:
            adata = temp_adata
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
        else:
            # x
            if adata.obsm['spatial'].shape[1] == 2:
                adata.obs['x'] = adata.obsm['spatial'][:, 0]
                adata.obs['y'] = adata.obsm['spatial'][:, 1]
                temp_adata.obs['x'] = temp_adata.obsm['spatial'][:, 0]
                temp_adata.obs['y'] = temp_adata.obsm['spatial'][:, 1]
            else:
                adata.obs['x'] = adata.obsm['spatial'][0, :]
                adata.obs['y'] = adata.obsm['spatial'][1, :]
                temp_adata.obs['x'] = temp_adata.obsm['spatial'][0, :]
                temp_adata.obs['y'] = temp_adata.obsm['spatial'][1, :]
            max_x_pr = adata.obs['x'].max()  #
            min_x_cu = temp_adata.obs['x'].min()  #
            delta_x = max_x_pr - min_x_cu + 10
            temp_adata.obs['x'] += delta_x
            # y
            mean_y_pr = adata.obs['y'].mean()  #
            mean_y_cu = temp_adata.obs['y'].mean()  #
            delta_y = mean_y_pr - mean_y_cu
            temp_adata.obs['y'] += delta_y
            # spatial
            temp_adata.obsm['spatial'] = np.vstack((temp_adata.obs['x'], temp_adata.obs['y'])).T
            # concatenate
            adata = adata.concatenate(temp_adata)
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
            adata.obs.fillna(0)
    print(adata)
    print(adata.shape)
    # (520815, 23761)
    adata = adata.copy()
    adata.obs['batch_period'] = adata.obs['batch_period'].astype('category')

def preprocess_3dspatial(adata):
    batch_col = 'batch'
    tmp = np.hstack([adata.obsm['spatial'], np.array([adata.obs[batch_col]]).T])
    tmp = pd.DataFrame(tmp)
    tmp.columns = ['x', 'y', 'batch']
    tmp.index = adata.obs_names
    tmp = tmp.astype(np.float32)

    delta_x = 0
    gap_size = 1000

    for i, x in enumerate(tmp['batch'].unique()):
        mu = tmp.loc[tmp['batch'] == x, ['x', 'y']].mean()
        tmp.loc[tmp['batch'] == x, ['x', 'y']] -= mu.astype('int')
        tmp.loc[tmp['batch'] == x, ['x']] += delta_x
        delta_x += gap_size
        # tmp.loc[tmp['batch']==x, ['y']] += delta_y[i]
    tmp -= tmp.min()
    tmp = tmp.astype(int)
    tmp = tmp.loc[adata.obs_names, ['x', 'y']].to_numpy()
    adata.obsm['spatial'] = tmp
    return adata


# Euclidean distance for ndarray
def euclidean_distances_ndarray(x):
    import ot
    dist = ot.dist(x)
    # print('Euclidean distance shape: ', dist.shape)
    return dist


def dist_2_adj_ndarray(dist):
    adj = np.where(dist != 0, 0.0, 1.0)
    return adj


# @torch.no_grad()
# def graph_construction_cluster(clusters, n_node, args=None, key=''):
#     # clusters = clusters.reshape(n_node, 1)
#     dist = euclidean_distances_ndarray(clusters)
#     adj = np.where(dist != 0, 0.0, 1.0)
#     adj =
#     # adj = torch.as_tensor(adj, dtype=torch.float32, device='cpu')
#     # plot_heatmap(adj, args, key)
#     return adj


@torch.no_grad()
def graph_construction_cluster(x):
    # x = x.reshape((n_node,1))
    dist = euclidean_distances_ndarray(x)
    # dist = dist_2_adj_ndarray(dist)
    dist = np.where(dist != 0, 0.0, 1.0)
    # plot_heatmap(adj, args, key)
    # dist = np.array(dist, dtype=np.float32)
    dist = dist.astype(np.float32)
    dist = ndarray2scipySparse(dist)
    return dist


def graph_construction(coor):
    import ot
    adj = ot.dist(coor)
    adj = 1 - adj / np.max(adj)
    # adj = sparse.csr_matrix(adj)
    adj, adj_norm, adj_ad = preprocess_graph(adj, coor.shape[0], True)
    graph_dict = {"adj_org": adj, "adj_norm": adj_norm, "adj_ad": adj_ad}
    gc.collect()
    return graph_dict


def graph_construction_gpu(adj_coo, cell_N):
    adata_Adj = graph_computing(adj_coo, cell_N)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    import networkx as nx
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    # adj_org = adj_org+ sparse.eye(adj_org.shape[0])

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 -= sparse.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    _, adj_norm_m1, _ = preprocess_graph(adj_m1, cell_N, diff=False)
    # adj_label_m1 = adj_m1 + sparse.eye(adj_m1.shape[0])
    # norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    if 'sparse' in str(type(adj_org)):
        adj_org = adj_org.toarray()
        adj_norm_m1 = adj_norm_m1.toarray()
    adj_org = torch.from_numpy(adj_org)
    adj_org = adj_org.cuda()
    adj_norm_m1 = torch.from_numpy(adj_norm_m1)
    adj_org = adj_org.cuda()
    graph_dict = {"adj_org": adj_org, "adj_norm": adj_norm_m1}

    return graph_dict


def graph_construction_3nn(coor, k_nn):
    import ot
    dist_matrix = ot.dist(coor, metric='euclidean')
    dist_matrix = torch.from_numpy(dist_matrix).cuda()
    _, idx1 = torch.sort(dist_matrix, descending=False)
    idx = idx1[:k_nn].cuda()
    dist_matrix = torch.zeros(dist_matrix.shape).cuda()
    dist_matrix = dist_matrix.scatter(1, idx, 1)
    dist_matrix.fill_diagonal_(0)
    # adj_org = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var)))
    adj_norm = normalize_adj_gpu(dist_matrix, True, True)
    adj_ad = diffusion_adj_gpu(dist_matrix, mode="ppr")
    # adj_label_m1 = adj_norm + torch.eye(adj_org.shape[0]).cuda()
    graph_dict = {"adj_org": dist_matrix.detach().cpu(), "adj_norm": adj_norm, "adj_ad": adj_ad}
    # ,"adj_label": adj_label_m1
    return graph_dict

# def graph_construction_3nn(X, k_nn):
#     import ot
#     dist_matrix = ot.sinkhorn(X)
#     dist_matrix = torch.from_numpy(dist_matrix).cuda()
#     _, idx1 = torch.sort(dist_matrix, descending=False)
#     idx = idx1[:k_nn].cuda()
#     dist_matrix = torch.zeros(dist_matrix.shape).cuda()
#     dist_matrix = dist_matrix.scatter(1, idx, 1)
#     dist_matrix.fill_diagonal_(0)
#     # adj_org = sparse.csr_matrix((data, (row, col)), shape=(len(obs), len(var)))
#     adj_norm = normalize_adj_gpu(dist_matrix, True, True)
#     adj_ad = diffusion_adj_gpu(dist_matrix, mode="ppr")
#     # adj_label_m1 = adj_norm + torch.eye(adj_org.shape[0]).cuda()
#     graph_dict = {"adj_org": dist_matrix.detach().cpu(), "adj_norm": adj_norm, "adj_ad": adj_ad}
#     # ,"adj_label": adj_label_m1
#     return graph_dict


# def adj_construction_3nn(coor, k_nn):
#     import ot
#     dist_matrix = ot.dist(coor, metric='euclidean')
#     dist_matrix = torch.from_numpy(dist_matrix).cuda()
#     _, idx1 = torch.sort(dist_matrix, descending=False)
#     idx = idx1[:k_nn].cuda()
#     dist_matrix = torch.zeros(dist_matrix.shape).cuda()
#     dist_matrix = dist_matrix.scatter(1, idx, 1)
#     dist_matrix.fill_diagonal_(0)
#     return dist_matrix
def adj_construction_3nn(coor, k_nn):
    import ot
    dist_matrix = ot.dist(coor, metric='euclidean')
    dist_matrix = torch.from_numpy(dist_matrix)
    _, idx1 = torch.sort(dist_matrix, descending=False)
    idx = idx1[:k_nn]
    dist_matrix = torch.zeros(dist_matrix.shape)
    dist_matrix = dist_matrix.scatter(1, idx, 1)
    dist_matrix.fill_diagonal_(0)
    return dist_matrix


def pos_eucliden_graph(coor, k_nn, weighted):
    from ot import dist
    num_nodes = coor.shape[0]
    dist_matrix = dist(coor, metric='euclidean').type(torch.float32)
    if weighted:
        idx = torch.sort(dist_matrix, descending=True)[1][:-k_nn].cuda()
        dist_matrix = 1.0 - dist_matrix.div(torch.max(dist_matrix)).cuda()
        dist_matrix = dist_matrix.scatter(1, idx, 0)
    else:
        idx = torch.sort(dist_matrix, descending=False)[1][:k_nn].cuda()
        # dist_matrix = torch.zeros((num_nodes, num_nodes), device=torch.device).scatter(1, idx, 1)
        dist_matrix = torch.zeros((num_nodes, num_nodes)).cuda()
        dist_matrix = dist_matrix.scatter(1, idx, 1)

    print('distance graph based on euclidean:')
    print('adj_org.shape: ', dist_matrix.shape, 'number of edges in adj_org:', torch.nonzero(dist_matrix).shape[0])
    return dist_matrix


@torch.no_grad()
def featsim_graph(feat, k_nn, weighted):
    print('Will the feature similarity weights of edges be added to the graph?', weighted)
    from ot import dist
    num_nodes = feat.shape[0]
    dist_matrix = dist(feat).type(torch.float32)
    if weighted:
        idx = torch.sort(dist_matrix, descending=True)[1][:-k_nn].cuda()
        dist_matrix = 1.0 - dist_matrix.div(torch.max(dist_matrix)).cuda()
        dist_matrix = dist_matrix.scatter(1, idx, 0)
    else:
        idx = torch.sort(dist_matrix, descending=False)[1][:k_nn].cuda()
        # dist_matrix = torch.zeros((num_nodes, num_nodes), device=torch.device).scatter(1, idx, 1)
        dist_matrix = torch.zeros((num_nodes, num_nodes))
        dist_matrix = dist_matrix.scatter(1, idx, 1)
    print('adj_org.shape: ', dist_matrix.shape, 'number of edges in adj_org:', torch.nonzero(dist_matrix).shape[0])
    return dist_matrix


@torch.no_grad()
def graph_construction_dist_featsim(feat, coor, k_nn, feat_sim, weighted):
    """
    :param feat: [num_nodes,num_genes]
    :param coor: [num_nodes,2]
    :return: adj_org[num_nodes,num_nodes], adj_ad[num_nodes,num_nodes]
    """

    print('Will feat_sim to be used? ', feat_sim, '. Will the weights of edges be added to the graph?', weighted)

    # position
    adj_org = pos_eucliden_graph(coor, k_nn, weighted)

    # feature similarity based on cosine_similarity
    if feat_sim:
        adj_org = adj_org.mm(featsim_graph(feat, k_nn, weighted))

    adj_org /= torch.max(adj_org)
    adj_org = adj_org.fill_diagonal_(0)

    # diffusion_adj
    adj_ad = diffusion_adj_gpu(adj_org)
    gc.collect()
    torch.cuda.empty_cache()
    print('constructed graph:')
    print('adj_org.shape: ', adj_org.shape, 'number of edges in adj_org:', torch.nonzero(adj_org).shape[0])
    adj_norm = normalize_adj_gpu(adj_org, True, True)
    # adj_ad = diffusion_adj_gpu(adj_org, mode="ppr", transport_rate=args.alpha_value)
    # adj_label_m1 = adj_norm + torch.eye(adj_org.shape[0]).cuda()
    graph_dict = {"adj_org": adj_org.detach().cpu(), "adj_norm": adj_norm, "adj_ad": adj_ad}
    return graph_dict


def graph_construction_spearmanr(feat, args, key=''):
    # features based correction with spearmanr
    import scipy
    adj = scipy.stats.mstats.spearmanr(feat)
    # plot_heatmap(adj, args, key)

    # k_nn
    adj = torch.from_numpy(adj)
    _, idx1 = torch.sort(adj, descending=True)
    idx = idx1[:args.k_nn]
    adj = torch.zeros(adj.shape)
    adj = adj.scatter(1, idx, 1)

    # self-loop
    adj.fill_diagonal_(1)
    return adj


@torch.no_grad()
def graph_construction_3nn_adj_org(coor, k_nn, key=''):
    import ot
    adj = ot.dist(coor, metric='euclidean')
    # if ifcuda:
    #     dist_matrix = torch.from_numpy(dist_matrix).cuda()
    # else:
    adj = torch.from_numpy(adj)
    _, idx1 = torch.sort(adj, descending=False)
    # if ifcuda:
    #     idx = idx1[:k_nn].cuda()
    #     adj_org = torch.zeros(dist_matrix.shape).cuda()
    # else:
    idx = idx1[:k_nn]
    adj = torch.zeros(adj.shape)
    adj = adj.scatter(1, idx, 1)
    adj.fill_diagonal_(1)
    adj = adj.to(torch.float32)
    adj = adj.numpy()
    adj = ndarray2scipySparse(adj)
    # # plot_heatmap(adj, args, key='spearmanr_adj')
    return adj

def graph_construction_spatial_knn(coor, k_nn):
    import ot
    adj = ot.dist(coor, metric='euclidean')
    # if ifcuda:
    #     dist_matrix = torch.from_numpy(dist_matrix).cuda()
    # else:
    adj = torch.from_numpy(adj)
    _, idx1 = torch.sort(adj, descending=False)
    # if ifcuda:
    #     idx = idx1[:k_nn].cuda()
    #     adj_org = torch.zeros(dist_matrix.shape).cuda()
    # else:
    idx = idx1[:k_nn]
    adj = torch.zeros(adj.shape)
    adj = adj.scatter(1, idx, 1)
    adj.fill_diagonal_(1)
    adj = adj.to(torch.float32)
    adj = adj.numpy()
    # adj = ndarray2scipySparse(adj)
    # # plot_heatmap(adj, args, key='spearmanr_adj')
    return adj
def graph_construction_ot(X, k_nn):
    import ot
    M = np.eye(X.shape[0])
    adj = ot.sinkhorn(X,X,M,1)
    # adj = (adj-np.min(adj))/(np.max(adj)-np.min(adj))
    # # adj = ndarray2scipySparse(adj)
    # # # plot_heatmap(adj, args, key='spearmanr_adj')
    adj = torch.from_numpy(adj)
    _, idx1 = torch.sort(adj, descending=False)
    # if ifcuda:
    #     idx = idx1[:k_nn].cuda()
    #     adj_org = torch.zeros(dist_matrix.shape).cuda()
    # else:
    idx = idx1[:k_nn]
    adj = torch.zeros(adj.shape)
    adj = adj.scatter(1, idx, 1)
    # adj.fill_diagonal_(1)
    adj = adj.to(torch.float32)
    adj = adj.numpy()
    return adj
def graph_construction_3nn_sparse(coor, k_nn):
    import ot
    # try:
    dist_matrix = ot.dist(coor, metric='euclidean')
    # except Exception as e:
    #     print(e)
    # print('ot.dist又出错了，利用sklearn方法计算')
    # dist_matrix = cosine_similarity(coor)
    # coo_matrix = coo_matrix(coo_matrix)
    dist_matrix = torch.from_numpy(dist_matrix).cuda()
    _, idx1 = torch.sort(dist_matrix, descending=False)
    idx = idx1[:k_nn].cuda()
    adj_org = torch.zeros(dist_matrix.shape).cuda()
    adj_org = adj_org.scatter(1, idx, 1)
    adj_org.fill_diagonal_(1)

    #  torch tensor -> torch sparse matrix
    # # adj_norm = normalize_adj_gpu(adj_org, True, True)
    # adj_ad = diffusion_adj_gpu(adj_org, mode="ppr", transport_rate=args.alpha_value)
    #
    # idx = torch.nonzero(adj_org).T  # 这里需要转置一下
    # data = adj_org[idx[0], idx[1]]
    # coo_adj_org = torch.sparse_coo_tensor(idx, data, adj_org.shape)
    #
    # idx = torch.nonzero(adj_ad).T  # 这里需要转置一下
    # data = adj_ad[idx[0], idx[1]]
    # coo_adj_ad = torch.sparse_coo_tensor(idx, data, adj_ad.shape)
    # return coo_adj_org, coo_adj_ad

    # scipy sparse -> torch sparse matrix
    adj_org, adj_norm, adj_ad = preprocess_graph(adj=adj_org, n_node=coor.shape[0], diff=True)
    graph_dict = {"adj_org": adj_org, "adj_norm": adj_norm, "adj_ad": adj_ad}
    gc.collect()
    return graph_dict


def graph_computing(adj_coo, cell_num, k_nn=10):
    from scipy.spatial import distance
    edgeList = []
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, 'euclidean')
        res = distMat.argsort()[:k_nn + 1]
        tmpdist = distMat[0, res[0][1:k_nn + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k_nn + 1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))
    return edgeList


@torch.no_grad()
def preprocess_graph(adj, n_node, diff=True, symmetry=False):
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    print('adj[0,0]:', adj[0, 0])

    # adj_norm
    adj_norm = normalize_adj(adj, symmetry=symmetry)
    assert adj.shape == adj_norm.shape

    # adj_ad
    if not diff:
        return adj, adj_norm, None
    else:
        # diffusion
        adj_ad = diffusion_adj(adj)
    # print('adj.shape: ', adj.shape)
    # print('adj_norm.shape: ', adj_norm.shape)
    # print('adj_ad.shape: ', adj_ad.shape)
    assert n_node == adj.shape[0]
    assert n_node == adj.shape[1]
    assert adj.shape == adj_norm.shape
    assert adj.shape == adj_ad.shape

    adj = torch.as_tensor(adj, dtype=torch.float32, device='cpu')
    adj_norm = torch.as_tensor(adj_norm, dtype=torch.float32, device='cpu')
    adj_ad = torch.as_tensor(adj_ad, dtype=torch.float32, device='cpu')
    graph_dict = {"adj_org": adj, "adj_norm": adj_norm, "adj_ad": adj_ad}
    return graph_dict


@torch.no_grad()
def preprocess_sparseG(adj, n_node, diff=True, symmetry=False):
    """
    adj: dense matrix <class 'numpy.ndarray'>

    """
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    print('adj.shape: ', adj.shape)
    assert n_node == adj.shape[0]
    assert n_node == adj.shape[1]
    # rowsum = np.array(adj.sum(1))
    # degree_mat_inv_sqrt = sparse.diags(np.power(rowsum, -0.5).flatten())
    print('adj[0,0]:', adj[0, 0])

    # adj_norm
    # adj_norm = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_norm = normalize_adj(adj)
    print('adj_norm.shape: ', adj_norm.shape)
    assert adj.shape == adj_norm.shape

    # adj_ad
    if not diff:
        # no diffusion
        # return adj, adj_norm, None
        adj_ad = None
    else:
        # diffusion
        adj_ad = diffusion_adj(adj)
        print('adj_ad.shape: ', adj_ad.shape)
        assert adj.shape == adj_ad.shape

    # adj = torch.as_tensor(adj, dtype=torch.float32, device='cpu')
    # # adj_norm = torch.cuda.FloatTensor(adj_norm)
    # adj_norm = torch.as_tensor(adj_norm, dtype=torch.float32, device='cpu')
    # # adj_ad = torch.cuda.FloatTensor(adj_ad)
    # adj_ad = torch.as_tensor(adj_ad, dtype=torch.float32, device='cpu')
    adj = torch.sparse.Tensor(adj.todense()) if 'sparse' in str(type(adj)) else torch.sparse.Tensor(adj)
    adj = adj.to_sparse()

    adj_norm = torch.sparse.Tensor(adj_norm.todense()) if 'sparse' in str(type(adj_norm)) else torch.sparse.Tensor(
        adj_norm)
    adj_norm = adj_norm.to_sparse()

    if not diff:
        graph_dict = {"adj_org": adj, "adj_norm": adj_norm}
    else:
        adj_ad = torch.sparse.Tensor(adj_ad.todense()) if 'sparse' in str(type(adj_ad)) else torch.sparse.Tensor(adj_ad)
        adj_ad = adj_ad.to_sparse()
        # adj = adj.cuda()

        graph_dict = {"adj_org": adj, "adj_norm": adj_norm, "adj_ad": adj_ad}
    return graph_dict  # return adj, adj_norm, adj_ad


def normalize_adj_gpu(adj, self_loop=True, symmetry=True):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        if adj.is_cuda:
            adj_1 = torch.eye(adj.shape[0]).cuda()
        else:
            adj_1 = torch.eye(adj.shape[0])
        adj_tmp = adj + adj_1
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)
    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = torch.matmul(d_inv, adj_tmp)
    return norm_adj


def normalize_adj(adj, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix < numpy.ndarray>
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if isinstance(adj,torch.Tensor):
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj.numpy()
    adj = adj.astype(np.float32)
    if adj[0, 0] == 0:
        adj = adj + np.eye(adj.shape[0])
    # calculate degree matrix and it's inverse matrix
    # degree = np.diag(np.array(adj.sum(0)).reshape((adj.shape[0])))
    # print('degree.shape: ', degree.shape)
    degree = np.array(adj.sum(0)).reshape((adj.shape[0]))
    print('degree.shape: ', degree.shape)
    degree = np.diag(degree)
    print('degree.shape: ', degree.shape)
    try:
        d_inv = np.linalg.inv(degree)
    except Exception as e:
        # 不可逆，求违逆
        d_inv = np.linalg.pinv(degree)
    print('d_inv.shape: ', d_inv.shape)
    # print()
    # print()
    # print()
    # print()
    # print()
    # print()
    #
    # print(d_inv)
    # print()
    # print()
    # print()
    #
    # print(adj)
    # print()
    # print()
    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        # sqrt_d_inv = np.sqrt(d_inv)
        # adj = np.matmul(np.matmul(sqrt_d_inv, adj), sqrt_d_inv)
        d_inv = np.sqrt(d_inv)
        adj = np.matmul(np.matmul(d_inv, adj), d_inv)
    # non-symmetry normalize: D^{-1} A
    else:
        adj = np.matmul(d_inv, adj)
    return adj


def diffusion_adj_gpu(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    # adj_tmp = adj + torch.eye(adj.shape[0]).cuda()
    adj.fill_diagonal_(1)
    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj.sum(0))
    d_inv = torch.linalg.inv(d)
    sqrt_d_inv = torch.sqrt(d_inv).cuda()

    # calculate norm adj
    norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # calculate graph diffusion
    # if mode == "ppr":
    diff_adj = transport_rate * torch.linalg.inv((torch.eye(d.shape[0]).cuda() - (1 - transport_rate) * norm_adj))

    return diff_adj


def diffusion_adj_tensor(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    # adj_tmp = adj + torch.eye(adj.shape[0])
    adj.fill_diagonal_(1)
    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj.sum(0))
    d_inv = torch.linalg.inv(d)
    sqrt_d_inv = torch.sqrt(d_inv)

    # calculate norm adj
    norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # calculate graph diffusion
    # if mode == "ppr":
    diff_adj = transport_rate * torch.linalg.inv((torch.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    if adj[0, 0] == 0:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    # d = np.diag(adj_tmp.sum(0))
    # d_inv = np.linalg.inv(d)
    d = np.array(adj_tmp.sum(0)).reshape((adj.shape[0]))
    d = np.diag(d)
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    # if mode == "ppr":
    diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def remove_edge_gpu(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        temp_sim_i = torch.argsort(similarity[i].cpu())[:int(round(remove_rate * n_node))]
        A[i, temp_sim_i] = 0
        A[temp_sim_i, i] = 0
    # normalize adj
    if 'tensor' in str(type(A)):
        Am = normalize_adj_gpu(A, self_loop=True, symmetry=True)
    else:
        Am = normalize_adj_gpu(numpy_to_torch(A).cuda(), self_loop=True, symmetry=True)
    # Am = numpy_to_torch(Am).cuda()
    return Am


def remove_edge(A, similarity, remove_rate=0.1):
    """
    remove edge based on embedding similarity
    Args:
        A: the origin adjacency matrix
        similarity: cosine similarity matrix of embedding
        remove_rate: the rate of removing linkage relation
    Returns:
        Am: edge-masked adjacency matrix
    """
    # remove edges based on cosine similarity of embedding
    n_node = A.shape[0]
    for i in range(n_node):
        temp_sim_i = torch.argsort(similarity[i])[:int(round(remove_rate * n_node))]
        A[i, temp_sim_i] = 0
        A[temp_sim_i, i] = 0

    # normalize adj
    # if 'tensor' in str(type(A)):
    # if A.is_cuda:
    #     Am = normalize_adj(A.cpu(), symmetry=True)
    # else:
    # else:
    # Am = normalize_adj(A, symmetry=True)
    A = A.numpy()
    Am = normalize_adj(A, symmetry=True)
    Am = numpy_to_torch(Am)

    return Am


def combine_graph_dict(dict_1, dict_2):
    # TODO add adj_org
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'], dict_2['adj_norm'])
    graph_dict = {"adj_norm": tmp_adj_norm, "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label'])}
    return graph_dict


def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def normalize_F(X):
    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X - mean) / std
    return X_norm


def preprocess_normalizeF(adata, args, data_name=None):
    if data_name is None:
        data_name = args.name
    labels = []

    # filter
    pp.filter_genes(adata, min_cells=1)
    pp.filter_cells(adata, min_genes=1)
    # # prefilter_specialgenes(adata)

    # Normalize
    print('\n------------------------------ preprocessed adata')
    # if 'sparse' in str(type(adata.X)):
    #     cnt_0 = np.count_nonzero(adata.X.toarray())
    # else:
    #     cnt_0 = np.count_nonzero(adata.X)
    # if cnt_0 * 2 < adata.shape[0] * adata.shape[1]:
    #     # pp.normalize_per_cell(adata)
    #     # pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    #     pp.normalize_total(adata, target_sum=1e4)
    #     if adata.X.max() > 10:
    #         pp.log1p(adata)
    #     pp.scale(adata)
    adata.X = normalize_F(adata.X)
    print(adata)
    print('\nx.max:', adata.X.max(), 'x.min:', adata.X.min())

    # labels
    for key in ['annotation', 'annotations', 'celltype_pred', 'cluster', 'clusters', 'Annotation',
                'celltype_mapped_refined', 'subclass']:
        if key in adata.obs:
            plot_adata(adata,key,args.res_dir,'annotation','png',args.name,True)
            labels = str2int_y(adata.obs[key].to_numpy(dtype=str))
            adata.obs['annotation'] = adata.obs[key]
    # PCA
    # if 'MB_E2' in data_name or 'X_pca' not in data_name:
    tl.pca(adata, svd_solver='arpack', n_comps=args.n_inputs)

    # leiden
    if 'leiden' not in adata.obs:
        pp.neighbors(adata, use_rep='X_pca')
        tl.leiden(adata, key_added="leiden")
    ari, fmi, nmi, acc, f1 = eva(labels, adata.obs['leiden'], args, modelname='leiden', data_name=data_name)
    if args.name == 'seqFISH':
        plt.rcParams['figure.figsize'] = 15, 4
        plt.rcParams['font.size'] = 10
        spot_size = 0.03
    else:
        spot_size = 30
    pl.spatial(adata, color=["leiden"], title=['Leiden (ARI=%.2f)' % ari], spot_size=spot_size, img=None, img_key=None,
               show=False)
    plt.savefig(osp.join(args.res_dir, data_name + '_leiden.pdf'), bbox_inches='tight')
    plt.close()

    # louvain
    if 'louvain' not in adata.obs:
        tl.louvain(adata)
    ari, fmi, nmi, acc, f1 = eva(labels, adata.obs['louvain'], args, modelname='louvain', data_name=data_name)
    pl.spatial(adata, color=["louvain"], title=['Louvain (ARI=%.2f)' % ari], spot_size=spot_size, img=None,
               img_key=None, show=False)
    plt.savefig(osp.join(args.res_dir, args.name + '_louvain.pdf'), bbox_inches='tight')
    plt.close()

    # kmeans
    model = KMeans(n_clusters=args.n_clusters, n_init=20)  # n_init = 20
    cluster_id = model.fit_predict(adata.obsm['X_pca'])
    adata.obs['kmeans'] = pd.Categorical(cluster_id)
    ari, fmi, nmi, acc, f1 = eva(labels, adata.obs['kmeans'], args, modelname='kmeans', data_name=data_name)
    pl.spatial(adata, color=["kmeans"], title=['KMeans (ARI=%.2f)' % ari], spot_size=spot_size, img=None, img_key=None,
               show=False)
    plt.savefig(osp.join(args.res_dir, args.name + '_kmeans.pdf'), bbox_inches='tight')
    plt.close()
    if 'sparse' in str(type(adata.obsm['X_pca'])):
        adata.obsm['X_pca'] = adata.obsm['X_pca'].toarray()
    return adata


def preprocess_nonormalize(adata, args, data_name=None):
    if data_name is None:
        data_name = args.name
    labels = []

    # filter
    # pp.filter_genes(adata, min_cells=1)
    # pp.filter_cells(adata, min_genes=1)
    # # prefilter_specialgenes(adata)

    # Normalize
    print('\n------------------------------ preprocessed adata')
    # if 'sparse' in str(type(adata.X)):
    #     cnt_0 = np.count_nonzero(adata.X.toarray())
    # else:
    #     cnt_0 = np.count_nonzero(adata.X)
    # if cnt_0 * 2 < adata.shape[0] * adata.shape[1]:
    #     # pp.normalize_per_cell(adata)
    #     # pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    #     pp.normalize_total(adata, target_sum=1e4)
    #     if adata.X.max() > 10:
    #         pp.log1p(adata)
    #     pp.scale(adata)
    # adata.X = normalize_F(adata.X)
    # print(adata)
    print('\nx.max:', adata.X.max(), 'x.min:', adata.X.min())

    # labels
    for key in ['annotation', 'annotations', 'celltype_pred', 'cluster', 'clusters', 'Annotation',
                'celltype_mapped_refined', 'subclass']:
        if key in adata.obs:
            plot_adata(adata,key,args.res_dir,'annotation','png',args.name,True)
            labels = str2int_y(adata.obs[key].to_numpy(dtype=str))
            adata.obs['annotation'] = adata.obs[key]
    # PCA
    # if 'MB_E2' in data_name or 'X_pca' not in data_name:
    tl.pca(adata, svd_solver='arpack', n_comps=args.n_inputs)
    adata.obsm['X_pca'] = StandardScaler().fit_transform(adata.obsm['X_pca'])

    # leiden
    if 'leiden' not in adata.obs:
        pp.neighbors(adata, use_rep='X_pca')
        tl.leiden(adata, key_added="leiden")
    leiden_preds = adata.obs['leiden'].values.codes
    ari, fmi, nmi, acc, f1 = eva(labels, leiden_preds, args, modelname='leiden', data_name=data_name)
    if args.name == 'seqFISH':
        plt.rcParams['figure.figsize'] = 15, 4
        plt.rcParams['font.size'] = 10
        spot_size = 0.03
    else:
        spot_size = 30
    pl.spatial(adata, color=["leiden"], title=['Leiden (ARI=%.2f)' % ari], spot_size=spot_size, img=None, img_key=None,
               show=False)
    plt.savefig(osp.join(args.res_dir, data_name + '_leiden.pdf'), bbox_inches='tight')
    plt.close()

    # louvain
    if 'louvain' not in adata.obs:
        tl.louvain(adata)
    louvain_preds = adata.obs['louvain'].values.codes
    ari, fmi, nmi, acc, f1 = eva(labels, louvain_preds, args, modelname='louvain', data_name=data_name)
    pl.spatial(adata, color=["louvain"], title=['Louvain (ARI=%.2f)' % ari], spot_size=spot_size, img=None,
               img_key=None, show=False)
    plt.savefig(osp.join(args.res_dir, args.name + '_louvain.pdf'), bbox_inches='tight')
    plt.close()

    # kmeans
    model = KMeans(n_clusters=args.n_clusters, n_init=20)  # n_init = 20
    cluster_id = model.fit_predict(adata.obsm['X_pca'])
    adata.obs['kmeans'] = pd.Categorical(cluster_id)
    ari, fmi, nmi, acc, f1 = eva(labels, cluster_id, args, modelname='kmeans', data_name=data_name)
    pl.spatial(adata, color=["kmeans"], title=['KMeans (ARI=%.2f)' % ari], spot_size=spot_size, img=None, img_key=None,
               show=False)
    plt.savefig(osp.join(args.res_dir, args.name + '_kmeans.pdf'), bbox_inches='tight')
    plt.close()
    if 'sparse' in str(type(adata.obsm['X_pca'])):
        adata.obsm['X_pca'] = adata.obsm['X_pca'].toarray()
    return adata


def preprocess(adata, args, data_name=None):
    # if data_name is None:
    #     data_name = args.name
    labels = []
    adata.var_names_make_unique()

    # filter
    pp.filter_genes(adata, min_cells=1)
    # prefilter_specialgenes(adata)
    pp.filter_cells(adata, min_genes=1)

    # Normalize
    if 'sparse' in str(type(adata.X)): cnt_0 = np.count_nonzero(adata.X.toarray())
    else: cnt_0 = np.count_nonzero(adata.X)
    if cnt_0 * 2 < adata.shape[0] * adata.shape[1]:
        # pp.normalize_per_cell(adata)
        # pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
        pp.normalize_total(adata, target_sum=1e4)
        if adata.X.max() > 10: pp.log1p(adata)
        pp.scale(adata)
    print('\n------------------------------ preprocessed adata')
    print(adata)
    print('\nx.max:', adata.X.max(), 'x.min:', adata.X.min())

    # labels
    for key in ['annotation_au', 'annotation', 'Annotation', 'annotations', 'celltype_pred', 'cluster', 'clusters', 'Annotation',
                'celltype_mapped_refined', 'subclass']:
        if key in adata.obs:
            plot_adata(adata=adata,key=key,res_dir=args.res_dir,modelname='annotation',file_type='png',data_name=args.name,if_plot_percluster=True)
            labels = str2int_y(adata.obs[key].values.codes)
            adata.obs['annotation'] = adata.obs[key]
            adata.obsm['annotation_num'] = labels
            args.n_clusters = len(np.unique(labels))
    # PCA
    # tl.pca(adata, svd_solver='arpack', n_comps=args.n_inputs)
    tl.pca(adata, n_comps=args.n_inputs)
    # StandardScaler
    # adata.obsm['X_pca'] = StandardScaler().fit_transform(adata.obsm['X_pca'])

    # leiden
    if 'sparse' in str(type(adata.obsm['X_pca'])):
        adata.obsm['X_pca'] = adata.obsm['X_pca'].toarray()
    cm = preprocess_3clustering(adata, 'X_pca', args)

    return adata, cm

def preprocess_dro(adata, args, data_name=None):
    # if data_name is None:
    #     data_name = args.name
    labels = []
    adata.var_names_make_unique()

    # filter
    pp.filter_genes(adata, min_cells=1)
    # prefilter_specialgenes(adata)
    pp.filter_cells(adata, min_genes=1)
    pp.scale(adata)
    tl.pca(adata)

    # labels
    for key in ['annotation_au', 'annotation', 'annotations', 'celltype_pred', 'cluster', 'clusters', 'Annotation',
                'celltype_mapped_refined', 'subclass']:
        if key in adata.obs:
            # plot_adata(adata, key, args)
            plot_adata(adata,key,args.res_dir,'annotation','png',args.name,True)
            labels = str2int_y(adata.obs[key].values.codes)
            adata.obs['annotation'] = adata.obs[key]
            adata.obsm['annotation_num'] = labels
            args.n_clusters = len(np.unique(labels))
    # StandardScaler
    # adata.obsm['X_pca'] = StandardScaler().fit_transform(adata.obsm['X_pca'])
    # preprocess_3clustering
    cm = preprocess_3clustering(adata, 'X_umap', args)
    if adata.obsm['X_umap'].shape[1] >= 30:
        adata.obsm['X_pca'] = adata.obsm['X_umap']
    if 'sparse' in str(type(adata.obsm['X_pca'])):
        adata.obsm['X_pca'] = adata.obsm['X_pca'].toarray()
    args.n_inputs = adata.obsm['X_pca'].shape[1]

    return adata, cm

def preprocess_3clustering(adata, key, args):
    # leiden
    if 'leiden' not in adata.obs:
        pp.neighbors(adata, use_rep=key)
        tl.leiden(adata, key_added="leiden")
    leiden_preds = adata.obs['leiden'].values.codes
    # louvain
    if 'louvain' not in adata.obs:
        tl.louvain(adata)
    louvain_preds = adata.obs['louvain'].values.codes
    # kmeans
    model = KMeans(n_clusters=args.n_clusters, n_init=20)  # n_init = 20
    cluster_id = model.fit_predict(adata.obsm[key])
    adata.obs['kmeans'] = pd.Categorical(cluster_id)
    if args.name == 'seqFISH':
        plt.rcParams['figure.figsize'] = 15, 4
        plt.rcParams['font.size'] = 10
        spot_size = 0.03
    elif '3D' in args.name or '_T' in args.name or 'Times' in args.name: spot_size = 1
    else: spot_size = 30
    # if 'annotation_num' in adata.obsm:
    # if 'celltype' in adata.obs:
    if 'annotation' in adata.obs:
        # labels = adata.obsm['annotation_num']
        from sklearn import preprocessing
        labels = adata.obs['annotation']
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        louvain_ari, fmi, nmi, acc, f1 = eva(labels, louvain_preds, args, modelname='louvain', data_name=args.name)
        print('Louvain:',louvain_ari, fmi, nmi, acc, f1)
        pl.spatial(adata, color=["louvain"], title=['Louvain (ARI=%.2f)' % louvain_ari], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_louvain.pdf'), bbox_inches='tight')
        plt.close()
        leiden_ari, fmi, nmi, acc, f1 = eva(labels, leiden_preds, args, modelname='leiden', data_name=args.name)
        print('Leiden:',louvain_ari, fmi, nmi, acc, f1)
        pl.spatial(adata, color=["leiden"], title=['Leiden (ARI=%.2f)' % leiden_ari], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_leiden.pdf'), bbox_inches='tight')
        plt.close()
        kmeans_ari, fmi, nmi, acc, f1 = eva(labels, cluster_id, args, modelname='kmeans', data_name=args.name)
        print('Kmeans:',louvain_ari, fmi, nmi, acc, f1)
        pl.spatial(adata, color=["kmeans"], title=['K-means (ARI=%.2f)' % kmeans_ari], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_kmeans.pdf'), bbox_inches='tight')
        plt.close()
        if louvain_ari == max(louvain_ari, leiden_ari, kmeans_ari):
            return 0
        elif leiden_ari == max(louvain_ari, leiden_ari, kmeans_ari):
            return 1
        else:
            return 2
    else:
        pl.spatial(adata, color=["louvain"], title=['Louvain'], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_louvain.pdf'), bbox_inches='tight')
        plt.close()
        pl.spatial(adata, color=["leiden"], title=['Leiden'], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_leiden.pdf'), bbox_inches='tight')
        plt.close()
        pl.spatial(adata, color=["kmeans"], title=['K-means'], spot_size=spot_size, img=None, img_key=None, show=False)
        plt.savefig(osp.join(args.res_dir, args.name + '_kmeans.pdf'), bbox_inches='tight')
        plt.close()
        return None

def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    i_adata.var_names_make_unique()
    pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = pp.scale(adata_X)
    adata_X = pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X


def prefilter_cells(adata, min_counts=None, max_counts=None, min_genes=200, max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[0], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_cells(adata.X, min_genes=min_genes)[0]) if min_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_cells(adata.X, max_genes=max_genes)[0]) if max_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_cells(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_cells(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw = pp.log1p(adata, copy=True)  # check the rowname
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:", adata.raw.var_names.is_unique)


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)


def preprocess_filter(adata):
    pp.filter_cells(adata, min_genes=1)
    # avoiding all genes are zeros
    prefilter_genes(adata, min_cells=1)
    prefilter_specialgenes(adata)

    if 'sparse' in str(type(adata.X)):
        adata.X = adata.X.toarray()
    if 'X_pca' in adata.obsm or 'n_count' in adata.obs or 'n_counts' in adata.obs:
        pass
    else:
        pp.normalize_total(adata, target_sum=1e4)
        pp.log1p(adata)
        pp.scale(adata)  # pp.pca(adata,n_comps=args.n_inputs)

    print(adata)
    return adata


def preprocess_2(adata):
    adatasub = adata.copy()
    adatasub.var_names_make_unique()
    if 'array' not in str(type(adatasub.X)):
        adatasub.X = np.exp(adatasub.X.toarray()) - 1
    # avoiding all genes are zeros
    prefilter_genes(adatasub, min_cells=3)
    prefilter_specialgenes(adatasub)
    # Normalize and take log for UMI
    pp.normalize_per_cell(adatasub)
    pp.log1p(adatasub)
    # print(adatasub)
    return adatasub


def str2int_y(labels):
    if not isinstance(labels[0], int):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
    else:
        pass
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype='int')
    return labels

def ndarray2scipySparse(ndarray):
    import scipy.sparse as sp
    sparse_mx = sp.csr_matrix(ndarray)
    sparse_mx.eliminate_zeros()
    return sparse_mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype('int'))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    sparse_mx_tensor = torch.sparse.FloatTensor(indices, values, shape)
    sparse_mx_tensor = sparse_mx_tensor
    return sparse_mx_tensor


def tensor_2_sparsetensor(matrix):
    # edge_index = nonzero(matrix).T
    # row = edge_index[0, :]
    # col = edge_index[1, :]
    # values = matrix[edge_index[0, :], edge_index[1, :]]
    # indices = torch.from_numpy(np.vstack((row, col)).astype('int'))
    # values = torch.from_numpy(values)
    # shape = torch.Size(matrix.shape)
    # sparse_matrix = torch.sparse.FloatTensor(indices, values, shape)
    if not matrix.is_sparse:
        sparse_matrix = matrix.to_sparse()
    # sparse_matrix.to_dense() 转换为稠密矩阵
    return sparse_matrix


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    # print(type(a))
    type_a = str(type(a))
    # if 'sparse' in type_a:
    #     sparse = True
    if 'tensor' in type_a or 'Tensor' in type_a:
        pass
    else:
        if sparse:
            if 'sparse' in type_a:
                a = torch.sparse.Tensor(a.todense())
            else:
                a = torch.sparse.Tensor(a)
            a = a.to_sparse()  # a = a.cuda()
        else:
            # a = torch.tensor(a)
            a = torch.as_tensor(a)  # .cuda()
    return a


def gaussian_noised_feature(X):
    """
    add gaussian noise to the attribute matrix X
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """

    N_1 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).cuda()
    N_2 = torch.Tensor(np.random.normal(1, 0.1, X.shape)).cuda()
    N_1 = X * N_1
    N_2 = X * N_2
    return N_1, N_2


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx, dtype=np.int32))


def load_pretrain_parameter(model, models_path, name):
    """
    load pretrained parameters
    Args:
        model: SGAE
    Returns: model
    """
    # pretrained_dict = torch.load(models_dir + name + '_pretrain.pkl',
    #                              map_location='cuda:{}'.format(torch.cuda.current_device()))
    pretrained_dict = torch.load(models_path, map_location='cuda:{}'.format(torch.cuda.current_device()))
    model_dict = model.state_dict()
    # if 'mouse_embryo' in args.name:
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     pretrained_dict = {k: v.T for k, v in pretrained_dict.items() if 'gae' in k}
    # else:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'gae' in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.eval()
    # if not model.iscuda:
    #     model = model.cuda()
    return model


def model_init(model, X, A_norm, n_clusters, models_path, name):
    """
    load the pre-train model and calculate similarity and cluster centers
    Args:
        model: SGAE
        X: input feature matrixn_clusters
        A_norm: normalized adj
    Returns: embedding similarity matrix
    """
    # load pre-train model
    model = load_pretrain_parameter(model, models_path, name)
    # model = model.eval()
    # print('load pre-train model: ')
    # print(model)

    # calculate embedding similarity
    # with torch.no_grad():
    #     _, _, _, sim, _, _, _, Z, _, _ = model(X, A_norm, X, A_norm)
    if not X.is_cuda: X = X.cuda()
    if not A_norm.is_cuda: A_norm = A_norm.cuda()
    _, _, _, sim, _, _, _, Z = model(X, A_norm, X, A_norm)

    # calculate cluster centers
    centers = clustering(Z, n_clusters)

    A_norm = A_norm.detach().cpu()
    Z = Z.detach().cpu()
    X = X.detach().cpu()

    del A_norm
    del model
    del Z
    del X
    gc.collect()
    torch.cuda.empty_cache()
    return sim, centers


def clustering_3methods(Z, adata, data_name, modelname, args):
    if isinstance(Z, torch.Tensor):
        if Z.is_cuda:
            Z = Z.detach().cpu()
        Z = Z.numpy()
    from pandas import Categorical
    try:
        # kmeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        preds_kmeans = kmeans.fit_predict(Z)
        np.savetxt(osp.join(args.res_dir, data_name + '_' + modelname + '_Z_kmeans_preds.txt'), preds_kmeans, fmt='%d')
        adata.obs['SGAE'] = Categorical(preds_kmeans)
        plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_kmeans', data_name=data_name,
                   if_plot_percluster=False)
        if 'annotation_num' in adata.obsm:
            eva_more_save(adata.obsm['annotation_num'], preds_kmeans, args.res_dir, args.name, args.gae_modelname + '_Z_kmeans')
        if 'celltype' in adata.obs:
            eva_more_save(adata.obs['celltype'], preds_kmeans, args.res_dir, args.name, args.gae_modelname + '_Z_kmeans')
        if 'annotation' in adata.obs:
            eva_more_save(adata.obs['annotation'], preds_kmeans, args.res_dir, args.name, args.gae_modelname + '_Z_kmeans')
    except:
        pass
    adata.obsm['embs'] = Z
    del Z
    gc.collect()
    pp.neighbors(adata, use_rep='embs')
    # tl.umap(adata)
    # min_nc = args.n_clusters - 10
    # max_nc = args.n_clusters + 10
    # for res_temp in range(min_nc, max_nc, 1):
    # resolut = res_temp * 0.1
    for resolut in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 2.5]:
        gc.collect()
        # louvain
        tl.louvain(adata, key_added="SGAE", resolution=resolut)
        # res, n_clusters_pred = res_search_fixed_clus('louvain', adata, resolut, increment=0.05, key='SGAE')
        plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_louvain_resolution_' + str(resolut),
                   data_name=data_name, file_type='png', if_plot_percluster=False)
        np.savetxt(osp.join(args.res_dir, data_name + '_' + modelname + '_Z_louvain_preds_resolution_' +
                            str(resolut) + '.txt'),
                   adata.obs['SGAE'].values.codes, fmt='%d')
        if 'annotation_num' in adata.obsm:
            eva_more_save(adata.obsm['annotation_num'], adata.obs['SGAE'].values.codes, args.res_dir, args.name,
                          args.gae_modelname + '_Z_louvain_resolution_' + str(resolut))
        if 'annotation' in adata.obsm:
            eva_more_save(adata.obsm['annotation'], adata.obs['SGAE'].values.codes, args.res_dir, args.name,
                          args.gae_modelname + '_Z_louvain_resolution_' + str(resolut))
        gc.collect()

        # # leiden
        # tl.leiden(adata, key_added="SGAE", resolution=resolut)
        # # res, n_clusters_pred = res_search_fixed_clus('louvain', adata, resolut, increment=0.05, key='SGAE')
        # # preds_louvain = adata.obs['SGAE'].values.codes
        # np.savetxt(osp.join(args.res_dir, data_name + '_' + modelname + '_Z_leiden_preds_resolution_'
        #                     + str(resolut) + '.txt'),
        #            adata.obs['SGAE'].values.codes, fmt='%d')
        # plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_leiden_resolution_' + str(resolut),
        #            data_name=data_name, file_type='png', if_plot_percluster=False)
        # if 'annotation_num' in adata.obsm:
        #     eva_more_save(adata.obsm['annotation_num'], adata.obs['SGAE'].values.codes, args.res_dir, args.name,
        #                   args.gae_modelname + '_Z_leiden_resolution_' + str(resolut))
        # gc.collect()
        # # return adata.obs['SGAE'].values.codes, adata.obs['SGAE'].values.codes, preds_kmeans


def clustering_MERFISH(modelname, Z, cell_batch_info, args):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    if 'tensor' in str(type(Z)) or 'Tensor' in str(type(Z)):
        Z1 = Z.data.cpu().numpy()
    else:
        Z1 = Z
    model = KMeans(n_clusters=args.n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z1)
    np.savetxt(args.preds_dir + args.name + '_' + modelname + '_pred_types.txt', cluster_id, fmt='%d', delimiter='\t')

    all_data = []
    for index in range(len(cluster_id)):
        all_data.append([index, cell_batch_info[index], cluster_id[index]])
    # noinspection PyTypeChecker
    np.savetxt(args.preds_dir + args.name + '_' + modelname + '_pred_types_batch.txt', np.array(all_data), fmt='%3d',
               delimiter='\t')

    return model.cluster_centers_


def clustering(Z, n_clusters):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    if 'tensor' in str(type(Z)) or 'Tensor' in str(type(Z)):
        Z = Z.detach().cpu().numpy()
    model = KMeans(n_clusters=n_clusters, n_init=20)  # 20
    # cluster_id = model.predict(Z)
    cluster_id = model.fit_predict(Z)
    return model.cluster_centers_


def clustering_mob(X, Z, n_clusters):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth
    Returns: acc, nmi, ari, f1, clustering centers
    Shorthand full name
    homo homogeneity score
    compl completeness score
    v-meas V measure
    ARI adjusted Rand index
    AMI adjusted mutual information
    silhouette silhouette coefficient

    """
    if isinstance(Z, torch.Tensor):
        if Z.is_cuda:
            Z = Z.data.detach().cpu()
        Z = Z.data.numpy()

    if X.is_cuda:
        X = X.data.detach().cpu().numpy()

    model = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z)

    inertia = model.inertia_
    chscore = calinski_harabasz_score(X, cluster_id)
    sscore = silhouette_score(X, cluster_id)
    dbscore = davies_bouldin_score(X, cluster_id)
    return chscore, sscore, dbscore, inertia, cluster_id

def clustering_4methods(Z, adata, data_name, modelname, args):
    if isinstance(Z, torch.Tensor):
        if Z.is_cuda:
            Z = Z.detach().cpu()
        Z = Z.numpy()
    pp.neighbors(adata, use_rep='embs')
    for resolut in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 2.5]:
        gc.collect()
        try:
            # louvain
            tl.louvain(adata, key_added="SGAE", resolution=resolut)
            # res, n_clusters_pred = res_search_fixed_clus('louvain', adata, resolut, increment=0.05, key='SGAE')
            plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_louvain_resolution_' + str(resolut),
                       data_name=data_name, file_type='png', if_plot_percluster=True)
            np.savetxt(osp.join(args.res_dir,
                                data_name + '_' + modelname + '_Z_louvain_preds_resolution_' + str(resolut) + '.txt'),
                       adata.obs['SGAE'].values.codes, fmt='%d')
            if 'annotation_num' in adata.obsm:
                eva_more_save(adata.obsm['annotation_num'], adata.obs['SGAE'].values.codes, args.res_dir, data_name,
                              modelname + '_Z_louvain_resolution_' + str(resolut))
            gc.collect()
        except:
            pass

        try:
            # leiden
            tl.leiden(adata, key_added="SGAE", resolution=resolut)
            # res, n_clusters_pred = res_search_fixed_clus('louvain', adata, resolut, increment=0.05, key='SGAE')
            # preds_louvain = adata.obs['SGAE'].values.codes
            np.savetxt(osp.join(args.res_dir,
                                data_name + '_' + modelname + '_Z_leiden_preds_resolution_' + str(resolut) + '.txt'),
                       adata.obs['SGAE'].values.codes, fmt='%d')
            plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_leiden_resolution_' + str(resolut),
                       data_name=data_name, file_type='png', if_plot_percluster=True)
            if 'annotation_num' in adata.obsm:
                eva_more_save(adata.obsm['annotation_num'], adata.obs['SGAE'].values.codes, args.res_dir, data_name,
                              modelname + '_Z_leiden_resolution_' + str(resolut))
            gc.collect()
            # return adata.obs['SGAE'].values.codes, adata.obs['SGAE'].values.codes, preds_kmeans
        except:
            pass

        try:
            # kmeans
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            preds_kmeans = kmeans.fit_predict(Z)
            np.savetxt(osp.join(args.res_dir, data_name + '_' + modelname + '_Z_kmeans_preds.txt'), preds_kmeans,
                       fmt='%d')
            adata.obs['SGAE'] = pd.Categorical(preds_kmeans)
            plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_Z_kmeans', data_name=data_name,
                       if_plot_percluster=True)
            if 'annotation_num' in adata.obsm:
                eva_more_save(adata.obsm['annotation_num'], preds_kmeans, args.res_dir, data_name,
                              modelname + '_Z_kmeans')
            adata.obsm['embs'] = Z
            del Z, preds_kmeans
            gc.collect()
        except:
            pass

        try:
            # DBSCAN
            import pandas as pd
            # preds_dbscan = DBSCAN(eps=0.1, min_samples=10, n_jobs=-1).fit_predict(Z)
            for min_samples in [2, 5, 10, 20, 50]:
                preds_dbscan = OPTICS(min_samples=min_samples).fit_predict(Z)
                np.savetxt(osp.join(args.res_dir,
                                    data_name + '_' + modelname + '_OPTICS_minsample' + str(
                                        min_samples) + '_preds.txt'),
                           preds_dbscan, fmt='%d')
                adata.obs['SGAE'] = pd.Categorical(preds_dbscan)
                plot_adata(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname + '_OPTICS_minsample' + str(min_samples), data_name=data_name,
                           if_plot_percluster=True)
                if 'annotation_num' in adata.obsm:
                    eva_more_save(adata.obsm['annotation_num'], preds_dbscan, args.res_dir, data_name, modelname + '_OPTICS_minsample' + str(min_samples))
        except:
            pass

def cluster_acc(y_true, y_pred):
    from munkres import Munkres
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    # print(y_true)
    # for i in y_true:
    #     print(i)
    y_true = y_true.astype('int')
    y_pred = y_pred.astype('int')
    y_true -= min(list(y_true))
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        # print('num_class1', num_class1)
        # print('num_class2', num_class2)
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        # print('num_class1', num_class1)
        # print('num_class2', num_class2)
        # print('num_class1 != numclass2')
        return -1, -1
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')

    return acc, f1_macro


def eva(y_true, y_pred, args=None, modelname='', show_details=True, data_name=None):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """

    if len(y_true) == 0:
        # print('no annotations !')
        return 0, 0, 0, 0, 0
    # print(y_pred.shape, y_pred)
    # print(y_true.shape, y_true)
    acc, f1 = cluster_acc(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    if show_details:
        print(' ARI {:.2f}'.format(ari), ', FMI {:.2f}'.format(fmi), ', NMI {:.2f}'.format(nmi), ', ACC {:.2f}'.format(acc), ', F1 {:.2f}'.format(f1))
    if data_name is None:
        data_name = args.name
    if args is not None:
        res_dir = args.res_dir
    else:
        res_dir = '../../'
    np.savetxt(osp.join(args.res_dir, data_name + '_' + modelname + '_res.txt'), [ari, fmi, nmi, acc, f1],
               delimiter='\t', fmt="%.5f")
    return ari, fmi, nmi, acc, f1


def cluster_acc_more(y_true, y_pred):
    from munkres import Munkres
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    # print(y_true)
    # for i in y_true:
    #     print(i)
    # y_true = y_true.astype('int')
    # y_pred = y_pred.astype('int')
    y_true = y_true - min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        # print('num_class1', num_class1)
        # print('num_class2', num_class2)
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        # print('num_class1', num_class1)
        # print('num_class2', num_class2)
        # print('num_class1 != numclass2')
        return -1, -1, -1, -1, -1, -1, -1
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')
    precision_macro = precision_score(y_true, new_predict, average='macro')
    recall_macro = recall_score(y_true, new_predict, average='macro')
    f1_micro = f1_score(y_true, new_predict, average='micro')
    precision_micro = precision_score(y_true, new_predict, average='micro')
    recall_micro = recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro


def eva_more_save(y_true, y_pred, res_dir, data_name, modelname):
    if y_true is None:
        print('no annotations !')
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    if len(y_true) == 0:
        print('no annotations !')
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    if isinstance(y_true[0], str):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y_true = le.fit_transform(y_true)
    # if not isinstance(y_true[0], int):
    #     y_true = np.array(y_true).astype('int')
    if not isinstance(y_pred[0], int):
        y_pred = np.array(y_pred).astype('int')

    acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = cluster_acc_more(y_true,
                                                                                                             y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    # if show_details:
    #     print(' ARI {:.2f}'.format(ari),
    #           ', FMI {:.2f}'.format(fmi),
    #           ', NMI {:.2f}'.format(nmi),
    #           ', ACC {:.2f}'.format(acc),
    #           ', F1 {:.2f}'.format(f1))
    np.savetxt(osp.join(res_dir, data_name + '_' + modelname + '_res.txt'),
               [ari, fmi, nmi, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro],
               delimiter='\t', fmt='%.5f')
    # print(data_name + '_' + modelname + '_res:\n', ari, fmi, nmi, acc, f1_macro, precision_macro, recall_macro,
          # f1_micro, precision_micro, recall_macro, recall_micro)
    return ari, fmi, nmi, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro


def eva_more(y_true, y_pred):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
    Returns:
    """
    if len(y_true) == 0:
        # print('no annotations !')
        return 0, 0, 0, 0, 0
    if isinstance(y_true[0], str):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y_true = le.fit_transform(y_true)

    # print(y_pred.shape, y_pred)
    # print(y_true.shape, y_true
    y_true = np.array(y_true).astype('int')
    y_pred = np.array(y_pred).astype('int')

    acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = cluster_acc_more(y_true,
                                                                                                             y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    # if show_details:
    #     print(' ARI {:.2f}'.format(ari),
    #           ', FMI {:.2f}'.format(fmi),
    #           ', NMI {:.2f}'.format(nmi),
    #           ', ACC {:.2f}'.format(acc),
    #           ', F1 {:.2f}'.format(f1))
    return ari, fmi, nmi, acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro




def plot_adata(adata, key, res_dir, modelname=None, file_type='pdf', data_name=None, if_plot_percluster=False):
    # # 绘图设置
    # # 文件首行设置 ‘Agg'
    # # import matplotlib
    # # matplotlib.use('Agg')
    # # dpi
    # import matplotlib.pyplot as plt
    # plt.rcParams['savefig.dpi'] = 500
    # spot_size

    # 把聚类类别设置为Categorical，图例才能是dots，否则是color bar
    if not is_categorical_dtype(adata.obs[key]):
        adata.obs[key] = pd.Categorical(adata.obs[key])
    if 'seqFISH' in data_name:
        plt.rcParams['figure.figsize'] = 15, 4
        plt.rcParams['font.size'] = 10
        spot_size = 0.03
    elif 'Liver' in data_name or 'ME_T'in data_name or 'ME_8'in data_name or 'times'in data_name or '3D' in data_name or 'Drosophila_3D' in data_name:
        spot_size = 1
    elif 'BRCA' in data_name:
        spot_size = 280
    else:
        spot_size = 30

    # 分群在空间位置上的分布
    # if len(np.unique(adata.obs[key].to_numpy())) < 90:
    #     pl.spatial(adata, color=key, spot_size=spot_size, img=None, img_key=None, show=False)
    #     # save
    #     if modelname is not None:
    #         save_path = osp.join(res_dir, data_name + '_' + modelname + '_preds' + '.' + file_type)
    #     else:
    #         save_path = osp.join(res_dir, data_name + '_annotation' + '.' + file_type)
    #     plt.savefig(save_path, bbox_inches='tight')
    #     plt.clf()
    #     plt.close()
    #
    #     # 每一个类别的分布
    #     if if_plot_percluster:
    #         cell_type_temp = np.unique(adata.obs[key])
    #         n_clusters_temp = len(cell_type_temp)
    #         for i in range(n_clusters_temp):
    #             g = cell_type_temp[i]
    #             pl.spatial(adata, color=[key], groups=[g], spot_size=spot_size, na_in_legend=False, show=False)
    #             # save
    #             g = str(g)
    #             if '/' in g: g = g.replace("/", "_")
    #             if modelname is not None: save_path = osp.join(res_dir, data_name + '_' + modelname + '_preds_' + g + '.' + file_type)
    #             else: save_path = osp.join(res_dir, data_name + '_annotation_' + g + '.' + file_type)
    #             plt.savefig(save_path, bbox_inches='tight')
    #             plt.close()


def plot_adata_eval(adata, key, res_dir, modelname=None, file_type='pdf', data_name=None, if_plot_percluster=False):
    plot_adata(adata=adata, key=key, res_dir=res_dir, modelname=modelname, file_type=file_type, data_name=data_name,
               if_plot_percluster=if_plot_percluster)
    if 'annotation_num' in adata.obsm:
        eva_more_save(y_true=adata.obsm['annotation_num'], y_pred=adata.obs[key].values.codes, res_dir=res_dir,
                      data_name=data_name, modelname=modelname)
    if 'annotation' in adata.obsm:
        eva_more_save(y_true=adata.obsm['annotation'], y_pred=adata.obs[key].values.codes, res_dir=res_dir,
                      data_name=data_name, modelname=modelname)
    if 'celltype' in adata.obsm:
        eva_more_save(y_true=adata.obsm['celltype'], y_pred=adata.obs[key].values.codes, res_dir=res_dir,
                      data_name=data_name, modelname=modelname)



def plot_adata_multi(adata, batch_key, key, data_name, modelname, res_dir, file_type='pdf', if_plot_percluster=True):

    # plot all and plot each cell type
    spot_size = 1

    plot_adata(adata=adata, key=key, res_dir=res_dir, modelname=modelname, file_type=file_type, data_name=data_name, if_plot_percluster=if_plot_percluster)

    # plot each batch_key
    # for spot_size in [1,3,10,20,30]:
    # plot each batch_key
    # period_temp = np.unique(adata.obs[batch_key])
    # n_period_temp = len(period_temp)
    # for i in range(n_period_temp):
    #
    #     g = period_temp[i]
    #     pl.spatial(adata, color=[key], groups=[g], spot_size=spot_size, na_in_legend=False, show=False)
    #     g = str(g)
    #     if '/' in g: g = g.replace("/", "_")
    #     if modelname is not None:
    #         save_path = osp.join(res_dir, data_name + '_period' + g + '_' + modelname + '_preds_spotsize'+str(spot_size)+'.'+ file_type)
    #     else:
    #         save_path = osp.join(res_dir, data_name + '_period' + g + '_annotation_spotsize'+str(spot_size)+'.'+ file_type)
    #     plt.savefig(save_path, bbox_inches='tight')
    #     plt.close()

def plot_eval_clustering(adata, labels2, modelname, emb_save_path, args):
    # if adata is not None:
    from pandas import Categorical
    adata.obs['SGAE'] = Categorical(labels2)
    plot_adata_eval(adata, key='SGAE', res_dir=args.res_dir, modelname=modelname, file_type='png', data_name=args.name,if_plot_percluster=True)
    import joblib
    Z = joblib.load(emb_save_path)
    # clustering_4methods(Z, adata, args.name, modelname, args)
    clustering_3methods(Z, adata, args.name, modelname, args)


def collate_fn(batch):
    batch = [x for x in zip(*batch)]
    ids, x, y, coor, adj_org, adj_norm, adj_ad, adj_am, X_tilde1, X_tilde2, idx = batch
    idx = torch.stack(idx).cuda()
    adj_org = torch.index_select(torch.stack(adj_org), 1, idx)
    adj_norm = torch.index_select(torch.stack(adj_norm), 1, idx)
    adj_ad = torch.index_select(torch.stack(adj_ad), 1, idx)
    adj_am = torch.index_select(torch.stack(adj_am), 1, idx)
    return (torch.stack(ids, 0), torch.stack(x, 0), torch.stack(y, 0), torch.stack(coor, 0), adj_org, adj_norm, adj_ad,
            adj_am, torch.stack(X_tilde1, 0), torch.stack(X_tilde2, 0), idx)

def find_files_in_curDIR_givenType_givenKeyword(files_path,key_word, file_type='*.h5ad'):
    import os
    g = os.walk(files_path)
    files = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            # print(os.path.join(path, file_name))
            if key_word in file_name:
                files.append(os.path.join(path, file_name))
    return files

def find_files_in_curDIRandsubDIR_givenType_givenKeyword(files_path, key_word, file_type='*.h5ad'):
    """找出目标文件"""
    print('files_path:',files_path)
    from pathlib import Path
    p = Path(files_path)
    fullname = []  # 存储指定类型所有文件名
    for file in p.rglob(file_type):  # 遍历指定文件夹所有指定类型文件
        fullname.append(str(file))
    files = []  # 所有目标文件名
    for i in fullname:
        print('file-', i)
        if key_word in i:
            files.append(i)
    return files
