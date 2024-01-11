'''
@Project : bias_correction_cfg.py 
@File    : cvt_grib_to_nc.py
@Author  : Wenyuan Li
@Date    : 2022/11/1 19:06 
@Desc    :  convert grib file to nc file
'''
import sys
sys.path.append(sys.path[0] + '/..')
import numpy as np
import xarray as xr
import os
import glob
import tqdm
from DeepPhysiNet.utils import path_utils
import multiprocessing
import argparse

def process_pressure(data_files,result_folder,thread_id=0):
    if thread_id==0:
        data_files=tqdm.tqdm(data_files)
    for data_file in data_files:
        file_name=path_utils.get_filename(data_file,is_suffix=False)
        for level in (1000,925,850,700,500):
            result_file = os.path.join(result_folder, '%s_%dhpa.nc' % (file_name,level))
            if os.path.exists(result_file):
                continue
            ds=xr.load_dataset(data_file,engine='cfgrib'
                               ,backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa','level':level}}
                               )
            ds.to_netcdf(result_file)
            idx_file=glob.glob(os.path.join(data_path,'%s*.idx'%file_name))
            for file in idx_file:
                os.remove(file)

def process_surface(data_files,result_folder,thread_id=0):
    if thread_id==0:
        data_files=tqdm.tqdm(data_files)
    for data_file in data_files:
        file_name=path_utils.get_filename(data_file,is_suffix=False)
        result_file = os.path.join(result_folder, '%s_surface.nc' % file_name)
        if os.path.exists(result_file):
            continue
        try:
            ds=xr.load_dataset(data_file,engine='cfgrib'
                               ,backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface','level':0}}
                               )
            ds.to_netcdf(result_file)

            result_file = os.path.join(result_folder, '%s_2m.nc' % file_name)
            if os.path.exists(result_file):
                continue
            ds = xr.load_dataset(data_file, engine='cfgrib'
                                 , backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}}
                                 )
            ds.to_netcdf(result_file)

            result_file = os.path.join(result_folder,'%s_10m.nc' % file_name)
            if os.path.exists(result_file):
                continue
            ds = xr.load_dataset(data_file, engine='cfgrib'
                                 , backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10}}
                                 )
            ds.to_netcdf(result_file)
        except:
            print(data_file)
            continue

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str,required=True)
    parse.add_argument('--result_path', type=str, required=True)
    parse.add_argument('--pressure', action='store_true',default=False)
    parse.add_argument('--num_threads', type=int, default=0)
    args=parse.parse_args()
    data_path = args.data_path
    result_folder = args.result_path
    pressure=args.pressure
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    data_files = glob.glob(os.path.join(data_path, '*.grib'))
    np.random.shuffle(data_files)
    num_thread = args.num_threads
    if pressure:
        process=process_pressure
    else:
        process=process_surface
    if num_thread<=0:
        process(data_files, result_folder, 0)
    else:
        train_data_process = multiprocessing.Pool(num_thread)
        for i in range(num_thread):
            start_idx = int(i * len(data_files) / num_thread)
            end_idx = int((i + 1) * len(data_files) / num_thread)
            train_data_process.apply_async(process, (data_files[start_idx:end_idx], result_folder, i))
        train_data_process.close()
        train_data_process.join()
