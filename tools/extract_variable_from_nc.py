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
import datetime
import glob
import os
import tqdm
import multiprocessing
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
import argparse

def extract_data(var_dict,var_name,index):
    data = var_dict[var_name][index]
    if len(data.shape) == 3:
        # data = data[:, 2:-2, 2:-2]
        data = data[:, ::-1]
    elif len(data.shape) == 2:
        # data = data[2:-2, 2:-2]
        data = data[::-1]
    else:
        raise NotImplementedError
    return data

options = ["TILED=TRUE", "COMPRESS=DEFLATE", "NUM_THREADS=8", "ZLEVEL=9"]

def process_pressure(data_files,data_path,result_folder,thread_id=0):
    var_name_list = ['u', 'v', 't', 'gh', 'q']
    proj_name_list = ['UU', 'VV', 'TT', 'GHT', 'QQ']
    ref_time=datetime.datetime(1970,1,1)
    if thread_id==0:
        pbar=tqdm.tqdm(range(0,len(data_files)*len(var_name_list)*30*2*61))
    for data_file in data_files:
        file_name = path_utils.get_filename(data_file, is_suffix=False)
        file_name = file_name.replace('_1000hpa', '')

        ncf_dataset_1000hpa = Dataset(data_file)
        var_dict_1000hpa = ncf_dataset_1000hpa.variables
        seconds = var_dict_1000hpa['time']
        data_len = len(seconds)
        step_list = var_dict_1000hpa['step']
        step_len = len(var_dict_1000hpa['step'])
        data_file = os.path.join(data_path, '%s_925hpa.nc' % file_name)
        var_dict_925hpa = Dataset(data_file).variables
        data_file = os.path.join(data_path, '%s_850hpa.nc' % file_name)
        var_dict_850hpa = Dataset(data_file).variables
        data_file = os.path.join(data_path, '%s_700hpa.nc' % file_name)
        var_dict_700hpa = Dataset(data_file).variables
        data_file = os.path.join(data_path, '%s_500hpa.nc' % file_name)
        var_dict_500hpa = Dataset(data_file).variables
        for var_name,proj_name in zip(var_name_list,proj_name_list):
            for i in range(data_len):
                second = seconds[i].data
                second = float(second)
                for step_i in range(step_len):
                    if thread_id == 0:
                        pbar.update(1)
                    step=step_list[step_i].data
                    time_stamp=ref_time+datetime.timedelta(seconds=second)
                    year=time_stamp.year
                    result_path = os.path.join(result_folder, '%04d' % year)
                    if not os.path.exists(result_path):
                        os.mkdir(result_path)
                    result_file = os.path.join(result_path,
                                               'GFS_%s_f%03d_%s.tiff' % (time_stamp.strftime('%Y-%m-%d-%H-%M-%S'),step,
                                                                      proj_name))
                    if os.path.exists(result_file):
                        continue
                    data_1000hpa= extract_data(var_dict_1000hpa,var_name,(i,step_i))
                    data_925hpa = extract_data(var_dict_925hpa, var_name, (i,step_i))
                    data_850hpa = extract_data(var_dict_850hpa, var_name, (i,step_i))
                    data_700hpa = extract_data(var_dict_700hpa, var_name, (i,step_i))
                    data_500hpa = extract_data(var_dict_500hpa, var_name, (i,step_i))
                    data=np.stack((data_1000hpa,data_925hpa,data_850hpa,data_700hpa,data_500hpa),axis=-3)
                    gdal_utils.save_full_image(result_file,data,options=options)

def process_surface(data_files,result_folder,var_name_list,proj_name_list,thread_id=0):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    ref_time=datetime.datetime(1970,1,1)

    for var_name,proj_name in zip(var_name_list,proj_name_list):
        if thread_id == 0:
            data_files = tqdm.tqdm(data_files,desc=var_name)
        for data_file in data_files:
            file_name=path_utils.get_filename(data_file,is_suffix=False)
            ncf_dataset=Dataset(data_file)
            var_dict=ncf_dataset.variables
            seconds=var_dict['time']
            data_len=len(seconds)
            step_list=var_dict['step']
            step_len=len(var_dict['step'])
            for i in range(data_len):
                second = seconds[i].data
                second = float(second)
                for step_i in range(step_len):
                    step=step_list[step_i].data
                    time_stamp=ref_time+datetime.timedelta(seconds=second)
                    year=time_stamp.year
                    result_path=os.path.join(result_folder,'%04d'%year)
                    # result_path=result_folder
                    if not os.path.exists(result_path):
                        os.mkdir(result_path)
                    result_file = os.path.join(result_path,
                                               'GFS_%s_f%03d_%s.tiff' % (time_stamp.strftime('%Y-%m-%d-%H-%M-%S'),step,
                                                                      proj_name))
                    if os.path.exists(result_file):
                        continue
                    data = var_dict[var_name][i,step_i]
                    if len(data.shape) == 3:
                        data = data[:,::-1]
                    elif len(data.shape) == 2:
                        data=data[::-1]
                    else:
                        raise NotImplementedError

                    if proj_name=='q2':
                        pres_file=os.path.join(result_path,
                                            'GFS_%s_f%03d_%s.tiff' % (time_stamp.strftime('%Y-%m-%d-%H-%M-%S'),step
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
    parse.add_argument('--pressure', action='store_true', default=False)
    parse.add_argument('--num_threads', type=int, default=0)
    args = parse.parse_args()
    print(args)
    data_path = args.data_path
    result_folder = args.result_path
    pressure = args.pressure
    num_thread = args.num_threads
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if pressure:
        data_files = glob.glob(os.path.join(data_path, '*_1000hpa.nc'))
        if num_thread <= 0:
            process_pressure(data_files,data_path, result_folder, 0)
        else:
            train_data_process = multiprocessing.Pool(num_thread)
            for i in range(num_thread):
                start_idx = int(i * len(data_files) / num_thread)
                end_idx = int((i + 1) * len(data_files) / num_thread)
                train_data_process.apply_async(process_pressure, (data_files[start_idx:end_idx], data_path,
                                                         result_folder,  i))
            train_data_process.close()
            train_data_process.join()
    else:
        data_files = glob.glob(os.path.join(data_path, '*10m.nc'))
        # np.random.shuffle(data_files)
        var_name_list = ['u10', 'v10', ]
        proj_name_list = ['u10', 'v10', ]
        if num_thread <= 0:
            process_surface(data_files, result_folder, var_name_list, proj_name_list,  0)
        else:
            train_data_process = multiprocessing.Pool(num_thread)
            for i in range(num_thread):
                start_idx = int(i * len(data_files) / num_thread)
                end_idx = int((i + 1) * len(data_files) / num_thread)
                train_data_process.apply_async(process_surface, (
                data_files[start_idx:end_idx], result_folder, var_name_list, proj_name_list, i))
            train_data_process.close()
            train_data_process.join()

        data_files = glob.glob(os.path.join(data_path, '*_surface.nc'))
        var_name_list = ['sp', ]
        proj_name_list = ['PSFC']
        if num_thread <= 0:
            process_surface(data_files, result_folder, var_name_list, proj_name_list, 0)
        else:
            train_data_process = multiprocessing.Pool(num_thread)
            for i in range(num_thread):
                start_idx = int(i * len(data_files) / num_thread)
                end_idx = int((i + 1) * len(data_files) / num_thread)
                train_data_process.apply_async(process_surface, (
                    data_files[start_idx:end_idx], result_folder, var_name_list, proj_name_list,  i))
            train_data_process.close()
            train_data_process.join()

        data_files = glob.glob(os.path.join(data_path, '*2m.nc'))
        # np.random.shuffle(data_files)
        var_name_list = ['t2m', 'd2m', ]
        proj_name_list = ['t2', 'q2', ]
        if num_thread <= 0:
            process_surface(data_files, result_folder, var_name_list, proj_name_list, 0)
        else:
            train_data_process = multiprocessing.Pool(num_thread)
            for i in range(num_thread):
                start_idx = int(i * len(data_files) / num_thread)
                end_idx = int((i + 1) * len(data_files) / num_thread)
                train_data_process.apply_async(process_surface, (
                    data_files[start_idx:end_idx], result_folder, var_name_list, proj_name_list, i))
            train_data_process.close()
            train_data_process.join()
