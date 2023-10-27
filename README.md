[![python >3.8.15](https://img.shields.io/badge/python-3.8.15-brightgreen)](https://www.python.org/)

# SGAE: Deciphering spatial domains from spatially resolved transcriptomics with Siamese Graph Autoencoder

## Overview
Spatial transcriptomics (ST) is a newly emerging field that facilitates a comprehensive characterization of tissue 
organization and architecture. By profiling the spatially-resolved gene expression patterns, ST technologies 
allow scientists to gain an in-depth understanding of the complex cellular dynamics and within tissue. 
Graph neural network (GNN) based methods usually suffer from representations collapse, which tends to map spatial spots 
into same representation. To address this issue, we proposed a Siamese Graph Autoencoder (SGAE) framework to learn 
discriminative spot representation and decipher accurate spatial domains. SGAE outperformed those spatial clustering 
methods across multiple platforms derived datasets based on the evaluation of ARI, FMI, NMI. Moreover, the clustering 
results derived from the SGAE model can be further utilized in 3D Drosophila Embryo reconstruction.
![](./fig1.png)

# Dependences
[![anndata-0.9.2](https://img.shields.io/badge/anndata-0.9.2-red)](https://github.com/scverse/anndata)
[![joblib-1.1.0](https://img.shields.io/badge/joblib-1.1.0-lightgrey)](https://pypi.org/project/joblib/1.0.1/)
[![matplotlib-3.5.1](https://img.shields.io/badge/matplotlib-3.5.1-brightgreen)](https://pypi.org/project/matplotlib/3.5.1/)
[![munkres-1.1.4](https://img.shields.io/badge/munkres-1.1.4-yellow)](https://pypi.org/project/munkres/)
[![networkx-2.6.3](https://img.shields.io/badge/networkx-2.6.3-green)](https://pypi.org/project/networkx/2.6.3/)
[![numpy-1.23.5](https://img.shields.io/badge/numpy-1.23.5-9cf)](https://pypi.org/project/numpy/1.23.5/)
[![pandas-1.5.3](https://img.shields.io/badge/pandas-1.5.3-informational)](https://pypi.org/project/pandas/1.5.3/)
[![pickle5-0.0.12](https://img.shields.io/badge/pickle5-0.0.12-1cf)](https://pypi.org/project/pickle5/)
[![POT-0.8.2](https://img.shields.io/badge/POT-0.8.2-orange)](https://pypi.org/project/POT/0.8.2/)
[![pynvml-11.5.0](https://img.shields.io/badge/pynvml-11.5.0-ff69b4)](https://pypi.org/project/pynvml/)
[![scanpy-1.9.4](https://img.shields.io/badge/scanpy-1.9.4-ff39b4)](https://pypi.org/project/scanpy/)
[![scikit_learn-1.0.2](https://img.shields.io/badge/scikit_learn-1.0.2-purple)](https://pypi.org/project/scikit-learn/1.0.2/)
[![scipy-1.6.2](https://img.shields.io/badge/scipy-1.6.2-cyan)](https://pypi.org/project/scipy/1.6.2/)
[![torch-2.0.1](https://img.shields.io/badge/torch-2.0.1-brigtblue)](https://pytorch.org/get-started/previous-versions/)
[![torch_geometric-2.3.1](https://img.shields.io/badge/torch_geometric-2.3.1-magenta)](https://pypi.org/project/torch-geometric/2.3.1/)
[![tqdm-4.59.0](https://img.shields.io/badge/tqdm-4.59.0-blueviolet)](https://pypi.org/project/tqdm/4.59.0/)

# Quick Start

### Install Dependencies via bash
```bash
### Python enviroment constructed by Conda
conda create -n SGAE python=3.8
conda activate SGAE
pip install -r requirements.txt
```


### Run on Code Ocean 

We also upload our code to [Code Ocean](https://codeocean.com/capsule/4678327/tree). Please check it for easier compilation.


### Running SGAE Script from the Command-line
```bash
cd SGAE
python3 run_sgae.py --n_epochs 1000 --name xxx  --data_file xxx.h5ad
```
Please specify your own data name and data_file via the arguments showed above. You can also
check the tutorial below to get a quick start.
# Data

We used data from various of platform and samples to benchmark our method. Here is a table for the data mentioned in article.

| Dataset | Platform |   Samples    |  Species   |             Tissue             |                            Source                            |
| :-----: | -------- | :----------: | :--------: | :----------------------------: | :----------------------------------------------------------: |
|  DLPFC  | 12       |  10X Visium  |   Human    | Dorsolateral prefrontal cortex |             http://research.libd.org/spatialLIBD             |
|   MG    | 1        |   seqFISH    |   Mouse    |          Gastrulation          |        https://crukci.shinyapps.io/SpatialMouseAtlas/        |
|   MC    | 1        |   MERFISH    |   Mouse    |          Cortex data           |              https://doi.brainimagelibrary.org/              |
|   MOB   | 1        | SLIDE-seq v2 |   Mouse    |         Olfactory bulb         | https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary |
|   DE    | 16       |  Stereo-seq  | Drosophila |             Embryo             |        https://db.cngb.org/stomics/flysta3d/spatial/         |
|   MB    | 1        |  Stereo-seq  |   Mouse    |             Brain              |              https://zenodo.org/record/7340795               |



# Codes

The foundation functions of SGAE is deposited at `models` directory. 

### Parameter settings

- General setting

  | Parameter   | Type | Defination                                         | Default |
  | ----------- | ---- | -------------------------------------------------- | ------- |
  | name        | str  | name to save result to indicate the data or sample | dblp    |
  | modelname   | str  | name to save result to indicate the model          | SGAE    |
  | project_dir | str  | directory to save result                           | ./      |
  | cuda        | bool | whether to use GPU                                 | True    |
  | gpu_id      | str  | choose a specific GPU                              | 0       |
  | seed        | int  | determine random seed                              | 1       |
  | n_clusters  | int  | number of clustering                               | 20      |

- Graph setting

  | Parameter   | Type  | Defination                             | Default |
  | ----------- | ----- | -------------------------------------- | ------- |
  | k_nn        | int   | number of neighbors to construct graph | 3       |
  | alpha_value | float | alpha value for graph diffusion        | 0.2     |

- Training setting

  | Parameter    | Type  | Defination                          | Default |
  | ------------ | ----- | ----------------------------------- | ------- |
  | n_epochs     | int   | total epoch                         | 1000    |
  | patience     | float | denote the early stopping point     | 0.2     |
  | batch_size   | int   | the size of a single batch          | 256     |
  | lr           | float | learning rate                       | 1e-4    |
  | lambda_value | float | weight for clustering guidance loss | 10      |



# Tutorial

 Reproduce the result of article via  `run_case.py`.

- DLPFC:
  ```python  
  python3 run_case.py --n_epochs 1000 --dataset dlpfc
  ```

- seqFISH  Mouse  Gastrulation:

  ```python  
  python3 run_case.py --n_epochs 1000 --dataset seqfish
  ```

- MERFISH  Mouse  Cortex data:

  ```python  
  python3 run_case.py --n_epochs 1000 --dataset merfish
  ```

- SLIDE-seq v2  Mouse  Olfactory bulb:

  ```python  
  python3 run_case.py --n_epochs 1000 --dataset slideseq
  ```

- Stereo-seq  Drosophila  Embryo:

  ```python  
  python3 run_case.py --n_epochs 1000 --dataset drosophila_14_16
  python3 run_case.py --n_epochs 1000 --dataset drosophila_16_18
  python3 run_case.py --n_epochs 1000 --dataset drosophila_l1
  ```

- Stereo-seq  Mouse  Brain:

  ```python  
  python3 run_case.py --n_epochs 1000 --dataset mousebrain
  ```


   

## Contact
Any questions or suggestions on EAGS are welcomed! Please report it on issues, 
or contact Lei Cao (caolei2@genomics.cn) or Shuangsang Fang (fangshuangsang@genomics.cn).
We recommend using [STOmics Cloud Platform](https://cloud.stomics.tech/) to access and use it.
