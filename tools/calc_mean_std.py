'''
@Project : bias_correction_cfg.py
@File    : calc_mean_std.py
@Author  : Wenyuan Li
@Date    : 2022/10/17 12:39
@Desc    :
'''
import sys
sys.path.append(sys.path[0] + '/..')
import numpy as np
import glob
import os
import datetime
import argparse
import tqdm
from DeepPhysiNet.utils import path_utils,gdal_utils
import multiprocessing

def process(data_path,var_names,result_path,thread_id=0):
    sub_step_total=5000
    sub_step_id=0
    for f in var_names:
        data_format=r'*_%s.tiff'%f
        result_mean_list=[]
        sub_result_list=[]
        data_files = glob.glob(os.path.join(data_path,'*', data_format))
        np.random.shuffle(data_files)
        data_files=data_files[::10]
        if thread_id==0:
            data_iter=tqdm.tqdm(data_files,desc='%s mean'%f)
        else:
            data_iter=data_files
        for data_file in data_iter:
            file_name=path_utils.get_filename(data_file,is_suffix=False)

            sub_step_id+=1
            img=gdal_utils.read_full_image(data_file,as_rgb=False,normalize=False,data_format='NUMPY_FORMAT')
            sub_result_list.append(img)
            if len(sub_result_list)>=sub_step_total:
                sub_result_list=np.stack(sub_result_list,axis=0)
                mean=np.mean(sub_result_list,axis=(0,1,2))
                result_mean_list.append(mean)
                sub_result_list=[]
        sub_result_list=np.stack(sub_result_list,axis=0)
        data_shape=gdal_utils.get_image_shape(data_file)
        total=0
        total_count=0
        for sub_mean in result_mean_list:
            total+=sub_mean*sub_step_total*data_shape[0]*data_shape[1]
            total_count=total_count+sub_step_total*data_shape[0]*data_shape[1]
        total=total+np.sum(sub_result_list,axis=(0,1,2))
        total_count=total_count+len(sub_result_list)*data_shape[0]*data_shape[1]

        mean=total/total_count
        # print(f,' mean:',mean.tolist())

        result_sigma_list=[]
        sub_result_list=[]
        if thread_id==0:
            data_iter=tqdm.tqdm(data_files,desc='%s std'%f)
        else:
            data_iter=data_files
        for data_file in data_iter:
            file_name = path_utils.get_filename(data_file, is_suffix=False)
            date_str = file_name.split('_')[1]
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')

            sub_step_id+=1
            img=gdal_utils.read_full_image(data_file,as_rgb=False,normalize=False,data_format='NUMPY_FORMAT')
            sub_result_list.append(img)
            if len(sub_result_list)>=sub_step_total:
                sub_result_list=np.stack(sub_result_list,axis=0)
                sigma=np.square(sub_result_list-mean)
                sigma=np.mean(sigma,axis=(0,1,2))
                result_sigma_list.append(sigma)
                sub_result_list=[]

        sub_result_list=np.stack(sub_result_list,axis=0)
        sub_result_list=np.square(sub_result_list-mean)

        total=0
        total_count=0
        for sub_mean in result_sigma_list:
            total+=sub_mean*sub_step_total*data_shape[0]*data_shape[1]
            total_count=total_count+sub_step_total*data_shape[0]*data_shape[1]
        total=total+np.sum(sub_result_list,axis=(0,1,2))
        total_count=total_count+len(sub_result_list)*data_shape[0]*data_shape[1]

        sigma=total/total_count
        std=np.sqrt(sigma)
        # print(f,' std:',std.tolist())
        result_file=os.path.join(result_path,'%s.txt'%f)
        with open(result_file,'w') as fp:
            fp.write('mean:{0};\n std:{1};'.format(mean.tolist(),std.tolist()))

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--result_path', type=str, required=True)
    parse.add_argument('--num_threads', type=int, default=0)
    args = parse.parse_args()
    data_path = args.data_path
    result_path = args.result_path
    num_thread = args.num_threads
    var_names=['PSFC','GHT','t2','TT','u10','UU','v10','VV','q2','QQ','rio']
    if num_thread <= 0:
        process(data_path,var_names,result_path,0)
    else:
        train_data_process = multiprocessing.Pool(num_thread)
        for i in range(num_thread):
            start_idx = int(i * len(var_names) / num_thread)
            end_idx = int((i + 1) * len(var_names) / num_thread)
            print('{0}: {1}'.format(i,var_names[start_idx:end_idx]))
            train_data_process.apply_async(process, (data_path,var_names[start_idx:end_idx],result_path,
                                                     i))
        train_data_process.close()
        train_data_process.join()