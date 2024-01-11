'''
@Project : deep-downscale 
@File    : calc_rio.py
@Author  : Wenyuan Li
@Date    : 2022/3/6 21:28 
@Desc    :  
'''
import sys
sys.path.append(sys.path[0] + '/..')
import glob
import os
import tqdm
from DeepPhysiNet.utils import gdal_utils,path_utils
import multiprocessing
import argparse

R_d=287.
options = ["TILED=TRUE", "COMPRESS=DEFLATE", "NUM_THREADS=8", "ZLEVEL=9"]
def process(data_files,thread_id=0):
    if thread_id==0:
        data_files=tqdm.tqdm(data_files)
    for data_file in data_files:
        file_name=path_utils.get_filename(data_file,is_suffix=False)
        file_name=file_name.replace('_PSFC','')
        p_path=path_utils.get_parent_folder(data_file,with_root=True)
        rio_file=os.path.join(p_path,'%s_rio.tiff'%file_name)
        if os.path.exists(rio_file):
            continue
        P=gdal_utils.read_full_image(data_file,as_rgb=False,normalize=False)[0]
        tmp_file=os.path.join(p_path,'%s_t2.tiff'%file_name)
        if not os.path.exists(tmp_file):
            continue
        T=gdal_utils.read_full_image(tmp_file,as_rgb=False,normalize=False)[0]
        q_file=os.path.join(p_path,'%s_q2.tiff'%file_name)
        q=gdal_utils.read_full_image(q_file,as_rgb=False,normalize=False)[0]

        R=(1+0.608*q)*R_d
        rio=P/R/T
        gdal_utils.save_full_image(rio_file,rio,options=options)


if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--num_threads', type=int, default=0)
    args = parse.parse_args()
    data_path = args.data_path
    data_files = glob.glob(os.path.join(data_path, '*/*_PSFC.tiff'))
    num_thread = args.num_threads

    if num_thread <= 0:
        process(data_files, 0)
    else:
        train_data_process = multiprocessing.Pool(num_thread)
        for i in range(num_thread):
            start_idx = int(i * len(data_files) / num_thread)
            end_idx = int((i + 1) * len(data_files) / num_thread)
            train_data_process.apply_async(process, (data_files[start_idx:end_idx], i))
        train_data_process.close()
        train_data_process.join()
