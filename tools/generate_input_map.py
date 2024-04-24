'''
@Project : deep-downscale 
@File    : filter_valid_date.py
@Author  : Wenyuan Li
@Date    : 2023/3/10 22:30 
@Desc    : 
'''
import pickle
import shutil

import numpy as np
import glob
import tqdm
import datetime
import os
from DeepPhysiNet.utils import path_utils
import argparse

if __name__=='__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, required=True)
    parse.add_argument('--result_file', type=str, required=True)
    parse.add_argument('--start_time', type=str,default='2007-01-01-00:00:00')
    parse.add_argument('--end_time', type=str,default='2020-12-31-12:00:00')

    args = parse.parse_args()
    print(args)
    data_path = args.data_path
    result_file=args.result_file
    data_files=glob.glob(os.path.join(data_path,'*/*.tiff'))
    query_dict={}
    for data_file in data_files:
        query_dict[path_utils.get_filename(data_file,is_suffix=False)]=data_file
    start_time=args.start_time
    end_time=args.end_time
    start_time=datetime.datetime.strptime(start_time,'%Y-%m-%d-%H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d-%H:%M:%S')

    days=(end_time-start_time).days*2
    step_list=list(range(0,361,6))
    variable_list =  ['PSFC','t2','q2','u10','v10','rio','UU','VV','TT','GHT','QQ']
    result_list=[]
    result_dict={}
    for day in tqdm.tqdm(range(days)):
        ref_time=start_time+datetime.timedelta(hours=day*12)
        date_str=ref_time.strftime('%Y-%m-%d-%H-%M-%S')
        try:
            for var_name in variable_list:
                for step in step_list:
                    file_name='GFS_%s_f%03d_%s'%(date_str,step,var_name)
                    if file_name not in query_dict.keys():
                        raise FileNotFoundError

        except FileNotFoundError:
            result_list.append(ref_time)
            continue

        for var_name in variable_list:
            for step in step_list:
                file_name = 'GFS_%s_f%03d_%s' % (date_str, step, var_name)
                if file_name in result_dict.keys():
                    raise KeyError()
                data_file=query_dict[file_name]
                p=path_utils.get_parent_folder(data_file,with_root=True)
                result_dict[file_name]=os.path.join(path_utils.get_parent_folder(p,with_root=False),
                                                path_utils.get_parent_folder(data_file, with_root=False),
                                                path_utils.get_filename(data_file,is_suffix=False))


    with open(result_file,'wb') as fp:
        pickle.dump(result_dict,fp)
