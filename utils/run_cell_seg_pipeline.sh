#!/usr/bin/bash
python_x='/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/cenweixuan/.conda/envs/deepcell/bin/python'
#taskset -c 3,4 $python_x /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/cenweixuan/cell_segmentation_v03/cell_seg_api.py -i data/mouse_embyro/E11.5_E1S3/FP200000554BL_A1_regist.tif -o data/mouse_embyro/E11.5_E1S3/FP200000554BL_A1_mask.tif
#taskset -c 3,4,5 $python_x cell_seg_api.py -i data/mouse_embyro/E11.5_E1S3/FP200000554BL_A1_regist.tif -o data/mouse_embyro/E11.5_E1S3/FP200000554BL_A1_mask.tif
#input_path='data/mouse_embyro/E10.5_E1S1/FP200000587TR_E4_20221209_103117_regist.tif'
#input_path='data/mouse_embyro/E15.5_E1S3/SS200000108BR_D1D2_20221213_141733_regist.tif'
#input_path='data/mouse_embyro/E15.5_E1S3/E15.5_E1S3_kidney_regist.tif'
input_path='/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/MB_fn/cellbin/B00713A2_SC_20221110_141827_1.2_regist.tif'
#taskset -c 3,4 nohup $python_x bin/cell_seg_api.py -i $input_path -o ${input_path/regist/mask} -th 5 > nohup.log 2>& 1&
#taskset -c 3,4 nohup $python_x -u bin/cell_seg_api_bak.py -i $input_path -o ${input_path/regist/mask} -g 5 > nohup.log 2>& 1&
taskset -c 203,204 nohup $python_x -u /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/cenweixuan/cell_segmentation_v03/cell_seg_api.py -i $input_path -o ${input_path/regist/mask} -g 5 > cell_seg.log 2>& 1&
