import anndata
import numpy as np
import pandas as pd
from scipy import sparse


def gem2h5ad(gem_path, binSize=50, Cellbin=True):
    '''
    transform a gem pd.DataFrame to a anndata object
    if Cellbin (default = True) is True and 'CellID' in gem, generate gem by Cellbin, otherwise use binSize
    '''
    replace_dic = {"values":"MIDCounts", 'UMICount': 'MIDCounts', 'MIDCount':'MIDCounts', 'UMICounts': 'MIDCounts',\
    'X':'x', 'Y':'y', 'cellID':'CellID'}
    if gem_path[-3:] == '.gz':
    	gem = pd.read_table(gem_path,comment='#', compression='gzip')
    else:
    	gem = pd.read_table(gem_path,comment='#')
    gem.columns = [replace_dic[x] if x in replace_dic else x for x in gem.columns]
    if ('CellID' in gem) and (Cellbin):
        gem['CellID'] = gem["CellID"].astype(int).astype('category')
        cell_list = gem["CellID"]
        gene_list = gem["geneID"].astype('category')
        data =  gem["MIDCounts"].to_numpy()
        row = cell_list.cat.codes.to_numpy()
        col = gene_list.cat.codes.to_numpy()
        obs = pd.DataFrame(index = cell_list.cat.categories)
        var = pd.DataFrame(index = gene_list.cat.categories)
        X = sparse.csr_matrix((data, (row, col)), shape = (len(obs), len(var)))
        adata = anndata.AnnData(X, obs = obs, var = var)
        spa = gem.groupby('CellID').median()
        adata.obsm['spatial'] = spa.loc[list(obs.index), ['x', 'y']].to_numpy()
        return adata
    else :
        gem['x'] = (gem['x']/int(binSize)).astype(np.uint32)
        gem['y'] = (gem['y']/int(binSize)).astype(np.uint32)
        gem['binID'] = (gem['x'].astype(str) + '-' + gem['y'].astype(str)).astype('category')
        cell_list = gem["binID"].astype('category')
        gene_list = gem["geneID"].astype('category')
        data = gem["MIDCounts"].to_numpy()
        row = cell_list.cat.codes.to_numpy()
        col = gene_list.cat.codes.to_numpy()
        obs = pd.DataFrame(index = cell_list.cat.categories)
        var = pd.DataFrame(index = gene_list.cat.categories)
        X = sparse.csr_matrix((data, (row, col)), shape = (len(obs), len(var)))
        adata = anndata.AnnData(X, obs = obs, var = var)
        adata.obsm['spatial'] = pd.Series(adata.obs.index).str.split('-', expand=True).astype('int').values
        #adata.raw = adata
        return adata