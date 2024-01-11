'''
@Project : deep-downscale 
@File    : physics_dataset.py
@Author  : Wenyuan Li
@Date    : 2022/2/11 16:39 
@Desc    :  Read the reanalysis data and observation data, and perform a unified normalization.
First, you need to convert the reanalysis data to TIFF format and save it separately according to the names
in the variable_cfg.
The saved filenames should follow a unified format: prefix_%Y-%m-%d_%H_%M_%S_varname.tiff
'''
import pickle
import torch
import numpy as np
import os
import glob
from DeepPhysiNet.utils import path_utils,gdal_utils
import torch.utils.data as data_utils
import re
import datetime
import tqdm
import xarray

class PhysicsDataset(data_utils.Dataset):
    def __init__(self,input_path,label_path,input_data_map_cfg,start_time,end_time,
                 input_variable_cfg:dict,
                 out_variable_cfg:dict,in_coord_file,out_coord_file,
                 constant_path,constant_variables=('landsea', 'elevation'),
                 input_time_step=6,input_time_step_nums=4,
                 label_time_step=1,label_img_size=64,dx=10800,dy=10800,
                 label_batch_size=1024,inter_batch_size=4096,
                 in_memory=True,obs_name_order=('u10','v10','pres','t2','q2','rio'),
                 forecast_time_period=360,start_fore_step=24,local_rank=0,**kwargs):
        '''
        :param input_path: The path of the input variables, which stores the low-resolution input reanalysis data with a time interval of `time_step`.
                           Note that the spatial resolution dx and dy are only related to the output.
        :param label_path: The path of the high-resolution reanalysis data, with the same file storage format as in input_path. The time interval of the data is `label_time_step`,
                           and the spatial interval is dx and dy.
        :param input_data_map: A dictionary that records the input variable names and paths.
        :param obs_path: The path of the observation data, which will be saved in the form of a pickle file after preprocessing.
        :param start_time: The start time of the data.
        :param end_time: The end time of the data. Note that the time here can be discontinuous, meaning there can be discontinuous times between endtime and starttime.
        :param input_variable_cfg: Records the input variable names, whether to automatically normalize, the normalization form, and normalization parameters.
                                   Each variable is saved in the form of a dictionary. For each variable, the above information is also saved as a dictionary.
        :param out_variable_cfg: Same as input_variable_cfg but for reanalysis data and observation data information. These data are considered as output data.
        :param out_coord_file: The coordinate file that records the output space, used to convert the longitude and latitude of observation data into projection coordinate values.
        :param input_time_step: The time interval of the input data, default: 6h.
        :param input_time_step_nums: The number of input data moments, default: 4. That is to say, under default conditions, input the current time data and 4 * input_time_step moments and the data in between.
        :param obs_time_step: The time interval of the observation data input.
        :param label_time_step: The time resolution of the high-resolution reanalysis data.
        :param label_img_size: The image size of the output data.
        :param dx: Spatial resolution.
        :param dy: Time resolution.
        :param label_batch_size: The batch size of reanalysis data, as it cannot be trained all at once.
        :param auto_norm: Whether to automatically calculate the normalization parameters for each variable.
        :param in_memory: Whether to preload all data into memory.
        :param obs_name_order: Since each observation data is stored as an array, the order must be maintained. This is used to indicate the position of each variable.
        :param forecast_time_period: The lead time of GFS forecast, default: 240h.
        :param start_fore_step: The interval of the start forecast time: 24h.
        :param kwargs: Additional arguments.
        '''
        super(PhysicsDataset, self).__init__()

        self.input_path=input_path
        self.label_path=label_path
        self.mode_names=[]
        if isinstance(input_data_map_cfg,dict):
            self.input_data_map={}
            for mode_key,file in input_data_map_cfg.items():
                with open(file,'rb') as fp:
                    data_map=pickle.load(fp)
                for file_key,img_file in data_map.items():
                    key='%s/%s'%(mode_key,file_key)
                    self.input_data_map[key]=img_file
                self.mode_names.append(mode_key)
        else:
            raise NotImplementedError

        self.start_time=datetime.datetime.strptime(start_time, '%Y-%m-%d_%H_%M_%S')
        self.end_time=datetime.datetime.strptime(end_time, '%Y-%m-%d_%H_%M_%S')
        self.input_variable_cfg = input_variable_cfg
        self.out_variable_cfg = out_variable_cfg
        self.out_coord_file=out_coord_file
        self.input_time_step=input_time_step
        self.input_time_step_nums=input_time_step_nums
        self.label_time_step=label_time_step
        self.label_img_size=label_img_size
        self.inter_batch_size=inter_batch_size
        self.dx=dx
        self.dy=dy
        self.input_data_dict={}
        self.label_data_dict={}
        self.obs_name_order=obs_name_order
        self.label_batch_size=label_batch_size
        self.in_momory = in_memory
        self.start_fore_step=start_fore_step
        self.forecast_time_period=forecast_time_period
        self.local_rank=local_rank
        with open(self.out_coord_file,'rb') as fp:
            lon,lat=pickle.load(fp)
        self.out_lon=lon[0]
        self.out_lat=lat[:,0]
        with open(in_coord_file,'rb') as fp:
            lon,lat=pickle.load(fp)
        self.in_lon=lon[0]
        self.in_lat=lat[:,0]
        assert self.in_lon[0]==self.out_lon[0] and self.in_lat[0]==self.out_lat[0]
        self.begin_lon=self.out_lon[0]
        self.begin_lat=self.out_lat[0]
        if self.local_rank==0:print('lon:{0},lat:{1}'.format(self.begin_lon,self.begin_lat))
        self.input_files,label_files=self.filter_input_files()
        if self.local_rank == 0:
            print('data files length: {0}'.format(len(self.input_files)))
            print('forecast data time span:{0}->{1} at {2} h'.format(path_utils.get_filename(self.input_files[0],is_suffix=False),
                                                   path_utils.get_filename(self.input_files[-1],is_suffix=False),
                                                self.start_fore_step))
            print('label data time span:{0}->{1}'.format(path_utils.get_filename(label_files[0], is_suffix=False),
                                                            path_utils.get_filename(label_files[-1], is_suffix=False
                                                                                    )))
        del label_files
        if isinstance(label_img_size, int) or isinstance(label_img_size, float):
            self.label_lat_size, self.label_lon_size = label_img_size, label_img_size
        elif (isinstance(label_img_size, list) or isinstance(label_img_size, tuple)) and len(label_img_size) == 2:
            self.label_lat_size, self.label_lon_size = label_img_size
        else:
            raise NotImplementedError
        assert len(self.out_lon)==self.label_lon_size and len(self.out_lat)==self.label_lat_size
        if self.local_rank==0:print('convert normal factor into array')
        for k,v in self.input_variable_cfg.items():
            self.input_variable_cfg[k]['norm_factor']=[np.array(v['norm_factor'][0]),np.array(v['norm_factor'][1])]
            dis_str='{0}:'.format(k)
            if self.local_rank==0:print(dis_str,self.input_variable_cfg[k]['norm_factor'])

        for key,v in self.out_variable_cfg.items():
            # self.out_variable_cfg[key]['norm_factor'] = [np.array(v['norm_factor'][0]),
            #                                              np.array(v['norm_factor'][1])]
            dis_str = '{0}:'.format(key)
            if self.local_rank==0:print(dis_str, self.out_variable_cfg[key]['norm_factor'])

        self.constant_variables=self.load_constant_variables(constant_path,constant_variables)

        self.has_normed=False


    def load_constant_variables(self,constant_path,constant_variable_names):
        constant_variables = []
        for name in constant_variable_names:
            data_file = os.path.join(constant_path, '%s.tiff' % name)
            data = gdal_utils.read_full_image(data_file, as_rgb=False, normalize=False)
            data = data.reshape(-1)
            constant_variables.append(data)
        constant_variables = np.array(constant_variables)
        min = np.min(constant_variables, axis=-1, keepdims=True)
        max = np.max(constant_variables, axis=-1, keepdims=True)
        constant_variables = (constant_variables - min) / (max - min)
        constant_variables = torch.from_numpy(constant_variables)
        return constant_variables


    def filter_input_files(self):
        result_list = []
        label_list=[]
        prefix=None
        var_name=None
        label_prefix=None
        label_var_name=None
        for mode_name in self.mode_names:
            start_time=self.start_time
            time_len=(self.end_time-self.start_time).total_seconds()/60//60/(self.start_fore_step)
            time_len=int(time_len)
            if self.local_rank==0:pbar=tqdm.tqdm(range(time_len),desc=mode_name)
            while start_time<=self.end_time:
                if self.local_rank==0:pbar.update(1)
                for p in range(0,self.forecast_time_period-self.input_time_step*self.input_time_step_nums+1,
                               self.input_time_step*self.input_time_step_nums):
                    obs_flag = True
                    for i in range(self.input_time_step_nums+1):
                        date_str=start_time.strftime('%Y-%m-%d-%H-%M-%S')
                        year_folder='%d'%start_time.year
                        ref_p=p+i*self.input_time_step
                        if prefix is None or var_name is None:
                            data_files=glob.glob(os.path.join(self.input_path,mode_name,year_folder,'*_%s_f%03d_*.tiff'%(date_str,ref_p)))
                            # assert len(data_files)>0
                            if len(data_files)==0:
                                break
                            data_file=data_files[0]
                            file_name = path_utils.get_filename(data_file, is_suffix=False)
                            key='%s/%s'%(mode_name,file_name)
                            if key not in self.input_data_map.keys():
                                break
                            data_file=os.path.join(self.input_path,'%s.tiff'%self.input_data_map[key])
                            m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
                            idx = m.regs[0][0]
                            prefix = file_name[0:idx]
                            date_str = m[0]
                            var_name=file_name[idx+len(date_str)+5:]
                        else:
                            tmp_key='%s/%s%s_f%03d%s'%(mode_name,prefix,date_str,ref_p,var_name)
                            if  tmp_key not in self.input_data_map.keys():
                                break
                            data_file=os.path.join(self.input_path,'%s.tiff'%self.input_data_map[tmp_key])
                        if not os.path.exists(data_file):
                            obs_flag=False
                            break
                    else:
                        if not obs_flag:
                            continue
                        label_time_step_num=self.input_time_step_nums*(self.input_time_step/self.label_time_step)
                        label_time_step_num=int(label_time_step_num)
                        sub_label_file_list=[]
                        for i in range(label_time_step_num+1):
                            ref_time = start_time + datetime.timedelta(hours=p+i * self.label_time_step)
                            date_str = ref_time.strftime('%Y-%m-%d-%H-%M-%S')
                            ref_p=p+i*self.label_time_step
                            if label_prefix is None or label_var_name is None:
                                data_files = glob.glob(os.path.join(self.label_path, '*_%s_*.tiff' % date_str))
                                if len(data_files)==0:
                                    continue
                                data_file = data_files[0]
                                file_name = path_utils.get_filename(data_file, is_suffix=False)
                                m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
                                idx = m.regs[0][0]
                                label_prefix = file_name[0:idx]
                                date_str = m[0]
                                label_var_name = file_name[idx + len(date_str):]
                            else:
                                data_file = os.path.join(self.label_path, '%s%s%s.tiff'%(label_prefix, date_str, label_var_name))
                            if os.path.exists(data_file):
                                sub_label_file_list.append(data_file)
                        if len(sub_label_file_list)==0:
                            obs_flag=False
                            continue
                        label_list.extend(sub_label_file_list)

                        date_str = start_time.strftime('%Y-%m-%d-%H-%M-%S')
                        year_folder = '%d' % start_time.year
                        tmp_key = '%s/%s%s_f%03d%s'%(mode_name,prefix,date_str,p,var_name)
                        if tmp_key not in self.input_data_map.keys():
                            break
                        # data_file = os.path.join(self.input_path,'%s.tiff'%self.input_data_map[tmp_key])
                        # data_file = os.path.join(self.input_path,year_folder, '%s%s_f%03d%s.tiff'%(prefix,date_str,p,var_name))
                        result_list.append('%s.tiff'%self.input_data_map[tmp_key])
                start_time=start_time+datetime.timedelta(hours=self.start_fore_step)
        result_list=list(set(result_list))
        result_list=sorted(result_list)
        label_list=list(set(label_list))
        label_list=sorted(label_list)
        return result_list,label_list

    def read_data(self,img_file,data_dict)->np.ndarray:
        file_name = path_utils.get_filename(img_file, is_suffix=False)
        if self.in_momory and file_name in data_dict.keys():
            img=data_dict[file_name]
            return img
        else:
            img = gdal_utils.read_full_image(img_file, as_rgb=False, normalize=False,data_format='NUMPY_FORMAT')
            return img

    def read_point(self,img_file,data_dict,x,y):
        file_name = path_utils.get_filename(img_file, is_suffix=False)
        if file_name not in data_dict.keys():
            img = gdal_utils.read_full_image(img_file, as_rgb=False, normalize=False, data_format='NUMPY_FORMAT')
            data_dict[file_name] = img

        data=data_dict[file_name][y,x]
        return data

    def __len__(self):
        return len(self.input_files)

    def norm_data(self,data,norm_factor,norm_type):
        if norm_type.lower()=='min_max':
            if isinstance(norm_factor,tuple) or isinstance(norm_factor,list):
                if len(norm_factor)==2:
                    min,max=norm_factor
                    data=(data-min)/(max-min)
                elif len(norm_factor)==1:
                    data = data / norm_factor[0]
                elif len(norm_factor)==3:
                    a_min,a_max,min=norm_factor
                    data=data-min
                    data=data**0.5
                    data = (data - a_min) / (a_max-a_min)
                else:
                    raise NotImplementedError
            else:
                data=data/norm_factor
        else:
            mean, std = norm_factor
            data = (data-mean)/std
        return  data

    def get_item_input(self,input_file):
        file_name = path_utils.get_filename(input_file, is_suffix=False)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        idx = m.regs[0][0]
        prefix = file_name[0:idx]
        date_str = m[0]
        forecast_str = file_name[idx + len(date_str) + 1:idx + len(date_str) + 5]
        forecast_h = int(forecast_str[1:])
        mode_name=path_utils.get_parent_folder(path_utils.get_parent_folder(input_file,with_root=True))
        field_data = []
        for date_id in range(self.input_time_step_nums+1):
            ref_p=forecast_h+self.input_time_step*date_id
            for key in self.input_variable_cfg.keys():
                var_dict = self.input_variable_cfg[key]
                var_name = var_dict['name']
                normal_factor = var_dict['norm_factor']
                use_norm = var_dict['use_norm']
                norm_type = var_dict['norm_type']
                key='%s/%s%s_f%03d_%s' % (mode_name,prefix, date_str,ref_p, var_name)
                input_file=os.path.join(self.input_path,'%s.tiff'%self.input_data_map[key])
                data = self.read_data(input_file,self.input_data_dict)
                if (not self.has_normed) and use_norm:
                    data = self.norm_data(data, normal_factor, norm_type)
                in_channels = data.shape[-1]
                data = data.reshape(-1, in_channels)
                data = np.transpose(data, (1, 0))
                field_data.append(data)
        field_data = np.concatenate(field_data, axis=0)
        field_data = torch.from_numpy(field_data)
        return field_data

    def get_item_label_data(self,input_file):
        source_input_file=input_file
        file_name = path_utils.get_filename(input_file, is_suffix=False)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        date_str = m[0]
        idx = m.regs[0][0]
        prefix = file_name[0:idx]
        forecast_str=file_name[idx+len(date_str)+1:idx+len(date_str)+5]
        forecast_h=int(forecast_str[1:])
        start_time = datetime.datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')+datetime.timedelta(hours=forecast_h)

        margin_x_rand = np.random.randint(0, self.label_lon_size, (self.label_batch_size,))
        margin_y_rand = np.random.randint(0, self.label_lat_size, (self.label_batch_size,))
        margin_lon_rand=self.begin_lon+margin_x_rand*0.25
        margin_lat_rand=self.begin_lat+margin_y_rand*0.25
        margin_t_rand = np.random.randint(0, self.input_time_step*self.input_time_step_nums+1, (self.label_batch_size,))
        margin_x_rand = margin_x_rand.tolist()
        margin_y_rand = margin_y_rand.tolist()
        margin_t_rand = margin_t_rand.tolist()
        margin_x = []
        margin_y = []
        margin_t = []
        margin_data = []
        label_dict = {}
        for x, y, t in zip(margin_x_rand, margin_y_rand, margin_t_rand):
            ref_time = start_time + datetime.timedelta(hours=t)
            date_str = ref_time.strftime('%Y-%m-%d-%H-%M-%S')
            item_results = []
            item_factor_results=[]

            for key in self.obs_name_order:
                var_dict = self.out_variable_cfg[key]
                var_name = var_dict['name']
                normal_factor = var_dict['norm_factor']
                use_norm = var_dict['use_norm']
                norm_type = var_dict['norm_type']
                input_file = os.path.join(self.label_path, '%s_%s_%s.tiff' % ('ERA5', date_str, var_name))
                data = self.read_point(input_file,label_dict,x,y)
                if (not self.has_normed) and use_norm:
                    data = self.norm_data(data, normal_factor, norm_type)
                item_results.append(data.tolist()[0])

            margin_data.append(item_results)
            margin_x.append(x)
            margin_y.append(y)
            margin_t.append((ref_time - start_time).total_seconds())



        inter_data=[]
        label_dict = {}
        file_name = path_utils.get_filename(source_input_file, is_suffix=False)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        date_str = m[0]
        idx = m.regs[0][0]
        prefix = file_name[0:idx]
        mode_name = path_utils.get_parent_folder(path_utils.get_parent_folder(source_input_file,with_root=True))
        forecast_str = file_name[idx + len(date_str) + 1:idx + len(date_str) + 5]
        forecast_h = int(forecast_str[1:])
        start_time = datetime.datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
        for key in self.obs_name_order:
            data_list = []
            var_dict = self.out_variable_cfg[key]
            var_name = var_dict['name']
            normal_factor = var_dict['norm_factor']
            use_norm = var_dict['use_norm']
            norm_type = var_dict['norm_type']
            for t in range(0, self.input_time_step * self.input_time_step_nums + 1, self.input_time_step):
                ref_p = forecast_h + t
                date_str = start_time.strftime('%Y-%m-%d-%H-%M-%S')
                key='%s/%s_%s_f%03d_%s' % (mode_name,'GFS', date_str, ref_p, var_name)
                input_file=os.path.join(self.input_path,'%s.tiff'%self.input_data_map[key])

                data = self.read_data(input_file, label_dict)
                if (not self.has_normed) and use_norm:
                    data = self.norm_data(data, normal_factor, norm_type)
                data_list.append(data)
            data = np.concatenate(data_list, axis=-1)
            y_len, x_len, t_len = data.shape
            assert len(self.in_lat)==y_len and len(self.in_lon)==x_len
            coord_t = np.array(range(0, t_len)) * self.input_time_step
            coord_x = self.in_lon
            coord_y = self.in_lat
            data = xarray.DataArray(data=data, dims=['y', 'x', 't'],
                                    coords=(
                                        coord_y.tolist(), coord_x.tolist(), coord_t.tolist()))
            var_list = data.interp(x=xarray.DataArray(margin_lon_rand, dims='z'),
                                   y=xarray.DataArray(margin_lat_rand, dims='z'),
                                   t=xarray.DataArray(margin_t_rand, dims='z'))

            inter_data.append(var_list.data)

        inter_data = np.stack(inter_data, axis=-1)
        margin_x = np.array(margin_x)
        margin_y = np.array(margin_y)
        margin_f = self.get_coriolis(margin_lat_rand)
        margin_f = torch.from_numpy(margin_f)
        margin_x = margin_x * self.dx
        margin_x = torch.from_numpy(margin_x)
        margin_y = margin_y * self.dy
        margin_y = torch.from_numpy(margin_y)
        margin_t = torch.from_numpy(np.array(margin_t))
        margin_data = np.array(margin_data)
        margin_data = torch.from_numpy(margin_data).float()
        inter_data = np.array(inter_data)
        inter_data = torch.from_numpy(inter_data).float()
        return  margin_x,margin_y,margin_t,margin_data,margin_f,inter_data

    def get_inter_data(self,input_file):
        file_name = path_utils.get_filename(input_file, is_suffix=False)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        date_str = m[0]
        idx = m.regs[0][0]
        prefix = file_name[0:idx]
        mode_name = path_utils.get_parent_folder(path_utils.get_parent_folder(input_file,with_root=True))
        forecast_str=file_name[idx+len(date_str)+1:idx+len(date_str)+5]
        forecast_h=int(forecast_str[1:])
        start_time = datetime.datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')

        inter_x_rand = np.random.rand(self.inter_batch_size)*(self.label_lon_size-1)
        inter_y_rand = np.random.rand(self.inter_batch_size)*(self.label_lat_size-1)
        inter_lon_rand = self.begin_lon + inter_x_rand * 0.25
        inter_lat_rand = self.begin_lat + inter_y_rand * 0.25
        inter_t_rand = np.random.randint(0,self.input_time_step*self.input_time_step_nums+1,(self.inter_batch_size,))
        inter_x_rand = inter_x_rand.tolist()
        inter_y_rand = inter_y_rand.tolist()
        inter_t_rand = inter_t_rand.tolist()

        inter_data = []
        label_dict = {}

        for key in self.obs_name_order:
            data_list=[]
            var_dict = self.out_variable_cfg[key]
            var_name = var_dict['name']
            normal_factor = var_dict['norm_factor']
            use_norm = var_dict['use_norm']
            norm_type = var_dict['norm_type']
            for t in range(0,self.input_time_step*self.input_time_step_nums+1,self.input_time_step):
                ref_p = forecast_h+t
                date_str = start_time.strftime('%Y-%m-%d-%H-%M-%S')
                key='%s/%s_%s_f%03d_%s' % (mode_name,'GFS', date_str, ref_p, var_name)
                input_file = os.path.join(self.input_path,'%s.tiff'%self.input_data_map[key])

                data = self.read_data(input_file,label_dict)
                if (not self.has_normed) and use_norm:
                    data = self.norm_data(data, normal_factor, norm_type)
                data_list.append(data)
            data=np.concatenate(data_list,axis=-1)
            y_len,x_len,t_len=data.shape
            assert len(self.in_lat) == y_len and len(self.in_lon) == x_len
            coord_t = np.array(range(0, t_len)) * self.input_time_step
            coord_x = self.in_lon
            coord_y = self.in_lat
            data = xarray.DataArray(data=data, dims=['y', 'x', 't'],
                                    coords=(
                                        coord_y.tolist(), coord_x.tolist(), coord_t.tolist()))
            var_list = data.interp(x=xarray.DataArray(inter_lon_rand, dims='z'),
                                   y=xarray.DataArray(inter_lat_rand, dims='z'),
                                   t=xarray.DataArray(inter_t_rand, dims='z'))

            inter_data.append(var_list.data)

        inter_data=np.stack(inter_data,axis=-1)

        inter_x = np.array(inter_x_rand)
        inter_y = np.array(inter_y_rand)
        inter_f = self.get_coriolis(inter_lat_rand)
        inter_f = torch.from_numpy(inter_f)
        inter_x = inter_x * self.dx
        inter_x = torch.from_numpy(inter_x)
        inter_y = inter_y * self.dy
        inter_y = torch.from_numpy(inter_y)
        inter_t = torch.from_numpy(np.array(inter_t_rand)*60*60)
        inter_data = np.array(inter_data)
        inter_data = torch.from_numpy(inter_data).float()
        return  inter_x,inter_y,inter_t,inter_data,inter_f

    def __getitem__(self, item):
        input_idx=item%len(self.input_files)
        input_file = self.input_files[input_idx]

        field_data=self.get_item_input(input_file)
        field_data=torch.cat((field_data,self.constant_variables),dim=0)
        margin_x, margin_y, margin_t, margin_data, margin_f,margin_input_data=self.get_item_label_data(input_file)
        inter_x,inter_y,inter_t,inter_data,inter_f=self.get_inter_data(input_file)
        file_name=path_utils.get_filename(input_file)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        date_str = m[0]
        idx = m.regs[0][0]
        forecast_str = file_name[idx + len(date_str) + 1:idx + len(date_str) + 5]
        forecast_h = int(forecast_str[1:])
        forecast_h=torch.tensor([forecast_h])#/self.forecast_time_period

        return field_data.float(),margin_x.float(),margin_y.float(),margin_t.float(),margin_data.float(),\
               margin_f.float(),margin_input_data.float(),inter_x.float(),inter_y.float(),inter_t.float(),\
               inter_data.float(),inter_f.float(),forecast_h.float(),input_file

    def get_coriolis(self,lat):
        omega = 7.29e-5
        f = 2*omega * np.sin(lat / 180 * np.pi)
        if len(f.shape)==1:
            f=np.expand_dims(f,axis=1)
        return f

    def get_margin_grid(self,input_file,margin_x_list,margin_y_list,margin_t_list):
        file_name = path_utils.get_filename(input_file, is_suffix=False)
        m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
        date_str = m[0]
        idx = m.regs[0][0]
        prefix = file_name[0:idx]
        mode_name = path_utils.get_parent_folder(path_utils.get_parent_folder(input_file,with_root=True))
        forecast_str = file_name[idx + len(date_str) + 1:idx + len(date_str) + 5]
        forecast_h = int(forecast_str[1:])
        start_time = datetime.datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
        label_dict={}
        inter_data=[]
        margin_lon_rand = self.begin_lon + np.array(margin_x_list) * 0.25
        margin_lat_rand = self.begin_lat + np.array(margin_y_list) * 0.25
        for key in self.obs_name_order:
            data_list = []
            var_dict = self.out_variable_cfg[key]
            var_name = var_dict['name']
            normal_factor = var_dict['norm_factor']
            use_norm = var_dict['use_norm']
            norm_type = var_dict['norm_type']
            for t in range(0, self.input_time_step * self.input_time_step_nums + 1, self.input_time_step):
                ref_p = forecast_h + t
                date_str = start_time.strftime('%Y-%m-%d-%H-%M-%S')
                key='%s/%s_%s_f%03d_%s' % (mode_name,'GFS', date_str, ref_p, var_name)
                input_file=os.path.join(self.input_path,'%s.tiff'%self.input_data_map[key])
                # input_file = os.path.join(self.input_path, '%s_%s_f%03d_%s.tiff' % ('GFS', date_str, ref_p, var_name))

                data = self.read_data(input_file, label_dict)
                if (not self.has_normed) and use_norm:
                    data = self.norm_data(data, normal_factor, norm_type)
                data_list.append(data)
            data = np.concatenate(data_list, axis=-1)
            y_len,x_len,t_len = data.shape
            assert len(self.in_lat) == y_len and len(self.in_lon) == x_len
            coord_t = np.array(range(0, t_len)) * self.input_time_step
            coord_x = self.in_lon
            coord_y = self.in_lat
            data = xarray.DataArray(data=data, dims=['y', 'x', 't'],
                                    coords=(
                                        coord_y.tolist(), coord_x.tolist(), coord_t.tolist()))

            var_list = data.interp(x=xarray.DataArray(margin_lon_rand, dims='z'),
                                   y=xarray.DataArray(margin_lat_rand, dims='z'),
                                   t=xarray.DataArray(margin_t_list, dims='z'))
            inter_data.append(var_list.data)

        inter_data = np.stack(inter_data, axis=-1)

        inter_x = np.array(margin_x_list)
        inter_y = np.array(margin_y_list)
        inter_f = self.get_coriolis(margin_lat_rand)
        inter_f = torch.from_numpy(inter_f)
        inter_x = inter_x * self.dx
        inter_x = torch.from_numpy(inter_x)
        inter_y = inter_y * self.dy
        inter_y = torch.from_numpy(inter_y)
        inter_t = torch.from_numpy(np.array(margin_t_list) * 60 * 60)
        inter_data = np.array(inter_data)
        inter_data = torch.from_numpy(inter_data).float()
        return inter_x, inter_y, inter_t, inter_data, inter_f



if __name__=='__main__':
    file_name=r'lekima201909_2019-08-03_00_00_00_GHT.tiff'
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2})", file_name)
    idx=m.regs[0][0]
    print(file_name[0:idx])
    print()