'''
@Project : deep-downscale 
@File    : extract_variable_from_GFS.py
@Author  : Wenyuan Li
@Date    : 2022/9/18 14:42 
@Desc    :  
'''
import sys
sys.path.append(sys.path[0] + '/..')
import numpy as np
from DeepPhysiNet.utils import  path_utils,gdal_utils,utils
from netCDF4 import Dataset,Variable
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import re
import datetime
import glob
import os
import tqdm
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import cv2
import argparse

import zipfile
import multiprocessing

options = ["TILED=TRUE", "COMPRESS=DEFLATE", "NUM_THREADS=8", "ZLEVEL=9"]

def process(data_files,result_path,var_name_list,proj_name_list,start_time,end_time,data_shape,thread_id=0):
    ref_time = datetime.datetime(1900, 1, 1)
    if thread_id==0:
        pbar=tqdm.tqdm(range(len(data_files)*365*2*24*5))
    for data_file in data_files:
        file_name = path_utils.get_filename(data_file, is_suffix=False)
        ncf_dataset = Dataset(data_file)
        var_dict = ncf_dataset.variables
        hours = var_dict['time']
        data_len = len(hours)
        for var_name,proj_name in zip(var_name_list,proj_name_list):
            for i in range(data_len):
                if thread_id==0:
                    pbar.update(1)
                hour = hours[i].data
                hour = float(hour)
                time_stamp=ref_time+datetime.timedelta(hours=hour)
                if not (time_stamp>=start_time and time_stamp<=end_time):
                    continue
                result_file = os.path.join(result_path,
                                           'ERA5_%s_%s.tiff' % (time_stamp.strftime('%Y-%m-%d-%H-%M-%S'),
                                                                  proj_name))
                if os.path.exists(result_file):
                    continue
                data = var_dict[var_name][i]
                if len(data.shape) == 3:
                    # data = data[:, 2:-2, 2:-2]
                    data = data[:,::-1]
                elif len(data.shape) == 2:
                    # data = data[2:-2, 2:-2]
                    data=data[::-1]
                else:
                    raise NotImplementedError
                assert data.shape[-1] == data_shape[-1] and data.shape[-2] == data_shape[-2]
                if proj_name=='q2':
                    pres_file=os.path.join(result_path,
                                        'ERA5_%s_%s.tiff' % (time_stamp.strftime('%Y-%m-%d-%H-%M-%S')
                                                                  ,'PSFC'))
                    pres_data=gdal_utils.read_full_image(pres_file,as_rgb=False,normalize=False)[0]
                    data=np.array(data)
                    data=specific_humidity_from_dewpoint(pres_data*units.pascal,data*units.kelvin)
                    data=np.array(data)
                gdal_utils.save_full_image(result_file,data,options=options)


if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--result_path', type=str, required=True)
    parse.add_argument('--num_threads', type=int, default=0)
    args = parse.parse_args()
    print(args)
    data_path = args.data_path
    result_path = args.result_path
    data_shape = (145, 257)
    var_name_list = ['t2m', 'sp', 'u10', 'v10', 'd2m']
    proj_name_list = ['t2', 'PSFC', 'u10', 'v10', 'q2']
    start_time = '2021-01-01-00:00:00'
    end_time = r'2022-12-31-23:00:00'
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d-%H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d-%H:%M:%S')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    data_files = glob.glob(os.path.join(data_path, '202*.nc'))
    np.random.shuffle(data_files)
    num_thread=args.num_threads
    if num_thread <= 0:
        process(data_files,result_path,var_name_list,proj_name_list,start_time,end_time,data_shape,0)
    else:
        train_data_process = multiprocessing.Pool(num_thread)
        for i in range(num_thread):
            start_idx = int(i * len(data_files) / num_thread)
            end_idx = int((i + 1) * len(data_files) / num_thread)
            print('{0}:{1}'.format(i,data_files[start_idx:end_idx]))
            train_data_process.apply_async(process, (data_files[start_idx:end_idx],result_path,var_name_list,
                                                     proj_name_list,start_time,end_time,data_shape, i))
        train_data_process.close()
        train_data_process.join()


