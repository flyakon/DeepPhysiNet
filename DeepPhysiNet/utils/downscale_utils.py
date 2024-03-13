'''
@Project : deep-downscale 
@File    : downscale_utils.py
@Author  : Wenyuan Li
@Date    : 2022/2/12 16:58 
@Desc    :  
'''

import numpy as np
from netCDF4 import Dataset,Variable
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from wrf import to_np, getvar, smooth2d, get_basemap, latlon_coords,projection
import xarray

import glob
import os
import tqdm
import pickle
import torch
import cv2
import torch.nn as nn

projection_dict={'Mercator':projection.Mercator,'LatLon':projection.LatLon}

def build_project(name='Mercator',**kwargs):
    if name in projection_dict.keys():
        return projection_dict[name](**kwargs)
    else:
        raise NotImplementedError('{0} is not an availabel value'.format(name))

class VisUtils(object):
    def __init__(self,coord_file,project_dict:dict,img_size=64):
        with open(coord_file,'rb') as fp:
            self.lon,self.lat=pickle.load(fp)

        if isinstance(img_size, int) or isinstance(img_size, float):
            img_size = (int(img_size), int(img_size))

        # self.lon=cv2.resize(self.lon,img_size)
        # self.lat=cv2.resize(self.lat,img_size)

        self.projection=build_project(**project_dict)

    def forward(self,data,result_file,show_img=True,show_cbar=True,**kwargs):
        if isinstance(data,torch.Tensor):
            data=data.detach().cpu().numpy()
            data=np.squeeze(data)
        plt.clf()

        data = xarray.DataArray(data=data, dims=["south_north", "west_east"],
                                coords=dict(
                                    XLAT=(["south_north", "west_east"], self.lat),
                                          XLONG=(["south_north", "west_east"], self.lon)),
                                attrs=dict(
                                    projection=self.projection)
                                )

        bm = get_basemap(data)
        # bm = Basemap(projection='cyl', llcrnrlat=self.lat[-1,0], urcrnrlat=self.lat[0,0], llcrnrlon=self.lon[0,0],
        #             urcrnrlon=self.lon[0,-1], resolution='i')
        bm.drawcoastlines(linewidth=0.25)
        bm.drawstates(linewidth=0.25)
        bm.drawcountries(linewidth=0.25)
        if not show_img:
            x, y = bm(to_np(self.lon), to_np(self.lat))
            bm.contour(x, y, to_np(data), 10, colors="black")
            bm.contourf(x, y, to_np(data), 10, cmap=get_cmap("jet"))
        else:
            bm.imshow(data,**kwargs)
        if show_cbar:
            plt.colorbar(shrink=.62)
        plt.tight_layout(pad=0)
        plt.savefig(result_file)


    def to_xy(self,ref_lon,ref_lat):
        tmp = (self.lon[0] - ref_lon) ** 2
        x1,x2 = np.argsort(tmp)[0:2]
        tmp = (self.lat[:, 0] - ref_lat) ** 2
        y1,y2 = np.argsort(tmp)[0:2]

        delta=(ref_lon-self.lon[0,x1])/(self.lon[0,x2]-self.lon[0,x1])
        x=(1-delta)*x1+delta*x2

        delta=(ref_lat-self.lat[y1,0])/(self.lat[y2,0]-self.lat[y1,0])
        y=(1-delta)*y1+delta*y2
        return x,y

class ProductsUtils(object):
    '''
    availabel variables include:
    u,v,w: verticity
    z: geo height or z_p1000
    T: temperature
    q: specific humidity
    rh: relative humidity
    rio: density of air

    slp: sea level pressure
    sst: surface temperature
    t2: 2m temperature
    td2: 2m dew point temperature
    u10m: 10m u-wind component
    v10m: 10m v-wind component
    u100m: 100m u-wind component
    u100m: 100m v-wind component
    tp: total precipitation during past 3 hours.
    '''
    def __init__(self,model:nn.Module,press_levels,img_size,pred_t_span,variables_dict:dict,
                 altitude_file,device='cuda:0'):
        self.model=model
        self.press_levels=press_levels
        self.device=device
        if isinstance(img_size, int) or isinstance(img_size, float):
            self.lat_size, self.lon_size = img_size, img_size
        elif (isinstance(img_size, list) or isinstance(img_size, tuple)) and len(img_size) == 2:
            self.lat_size, self.lon_size = img_size
        else:
            raise NotImplementedError

        with open(altitude_file,'rb') as fp:
            self.altitude=pickle.load(fp)
            self.altitude=cv2.resize(self.altitude,(self.lon_size,self.lat_size))

        self.pred_t_span=pred_t_span
        self.variable_dict=variables_dict
        self.func_dict={}
        # self.__available_variables=['u','v','w','z','T','q','rh','rio',
        #                             'slp','sst','t2','td2','u10m','v10m',
        #                             'u100m','v100m','tp']
        self.__available_variables = ['u','v','w','z','T','q',
                                      'rh','rio',
                                      'slp','sst','t2','td2','rh2','u10m','v10m','u100m','v100m','wd10m','wd',
                                      'tp']

        for v_name in self.__available_variables:
            if v_name in ['u','v','w','z','T','q']:
                func_name='forward_basic_variable'
                v_name='basic_variable'
            else:
                func_name='forward_%s'%v_name
            self.func_dict[v_name]=getattr(self,func_name)

        self.intermidate_variables_dict = {}

    def check_available(self,opt_variables):
        for name in opt_variables:
            tmp=name.split('_')
            if not (len(tmp)==2 or len(tmp)==1):
                raise Exception('{0} format not support.'.format(name))
            varname=tmp[0]
            if varname not in self.__available_variables:
                raise NotImplementedError('{0} not support.'.format(name))

    def forward(self,x,y,p,t,opt_variables:list,**kwargs):
        self.check_available(opt_variables)

        x=np.array(x)
        y=np.array(y)
        p=np.array(p)
        t=np.array(t)

        input_x=x/self.lon_size
        input_y=y/self.lat_size
        input_p=(p-self.press_levels[-1])/(self.press_levels[0]-self.press_levels[-1])
        input_t=t/self.pred_t_span

        input_x=torch.from_numpy(input_x).float().to(self.device)
        input_y=torch.from_numpy(input_y).float().to(self.device)
        input_p=torch.from_numpy(input_p).float().to(self.device)
        input_t=torch.from_numpy(input_t).float().to(self.device)
        input_samples=torch.stack([input_x,input_y,input_p,input_t],dim=1)

        result_dict={}
        self.intermidate_variables_dict.clear()

        for name in opt_variables:
            tmp=name.split('_')
            if len(tmp)==2:
                var_name,params=tmp
            else:
                var_name,params=tmp[0],None

            if name not in self.intermidate_variables_dict.keys():
                if var_name in ['u','v','w','z','T','q']:
                    result_dict[name]=self.func_dict['basic_variable'](var_name,input_samples,x,y,p,t,params=params)
                else:
                    result_dict[name]=self.func_dict[var_name](input_samples,x,y,p,t,params=params)
            else:
                result_dict[name]=self.intermidate_variables_dict[name]

            self.intermidate_variables_dict[name] = result_dict[name]

        self.intermidate_variables_dict.clear()

        return result_dict


    def forward_basic_variable(self,var_name,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            data=self.gather_variable(var_name,input_samples,x,y,p,t)



        if 'params' in kwargs.keys() and kwargs['params'] is not None:
            params=kwargs['params']
            p=params[1:]
            p=float(p)*100
            p_id=self.press_levels.index(p)
            data=data[p_id]
            data = smooth2d(data,3, cenweight=4)

        return data

    def forward_rh(self,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            T=self.gather_variable('T',input_samples,x,y,p,t)
            q = self.gather_variable('q', input_samples, x, y, p, t)

        press_data = np.array(self.press_levels)
        press_data = np.reshape(press_data, (-1, 1, 1))

        e=q*press_data/0.622
        t=T-273.15
        E=611.2*np.exp(17.67*t/(t+243.5))
        rh=e*100/E

        if 'params' in kwargs.keys() and kwargs['params'] is not None:
            params=kwargs['params']
            p=params[1:]
            p=float(p)*100
            p_id=self.press_levels.index(p)
            rh=rh[p_id]
            rh = smooth2d(rh,3, cenweight=4)
        return rh

    def forward_rio(self,input_samples,x,y,p,t,**kwargs):
        R_d=287
        with torch.no_grad():
            T=self.gather_variable('T',input_samples,x,y,p,t)
            q = self.gather_variable('q', input_samples, x, y, p, t)

        press_data = np.array(self.press_levels)
        press_data = np.reshape(press_data, (-1, 1, 1))

        rio=press_data/(1+0.608*q)/T/R_d

        if 'params' in kwargs.keys():
            params=kwargs['params']
            p=params[1:]
            p=float(p)*100
            p_id=self.press_levels.index(p)
            rio=rio[p_id]
            rio = smooth2d(rio,3, cenweight=4)
        return rio


    def forward_wd(self,input_samples,x,y,p,t,**kwargs):
        R_d=287
        with torch.no_grad():
            u=self.gather_variable('u',input_samples,x,y,p,t)
            v = self.gather_variable('v', input_samples, x, y, p, t)
            w = self.gather_variable('w', input_samples, x, y, p, t)

        press_data = np.array(self.press_levels)
        press_data = np.reshape(press_data, (-1, 1, 1))

        wd=u**2+v**2+w**2
        wd=wd**0.5

        if 'params' in kwargs.keys():
            params=kwargs['params']
            p=params[1:]
            p=float(p)*100
            p_id=self.press_levels.index(p)
            wd=wd[p_id]
            wd = smooth2d(wd,3, cenweight=4)
        return wd

    def forward_slp(self,input_samples,x,y,p,t,**kwargs):

        with torch.no_grad():
            z=self.gather_variable('z',input_samples,x,y,p,t)
        press_data=np.array(self.press_levels)
        press_data=np.reshape(press_data,(-1,1,1))
        slp=self.interp_z(0,z,press_data,selected_levels=(0,1,2,3,4))
        slp = smooth2d(slp,3, cenweight=4)
        return slp

    def forward_sst(self,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            z=self.gather_variable('z',input_samples,x,y,p,t)
            T = self.gather_variable('T', input_samples, x, y, p, t)

        sst=self.interp_z(0,z,T,selected_levels=(0,1,2,3,4))
        sst = smooth2d(sst,3, cenweight=4)
        return sst

    def forward_t2(self,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            z=self.gather_variable('z',input_samples,x,y,p,t)
            T=self.gather_variable('T',input_samples,x,y,p,t)

        height=self.altitude+2
        t2=self.interp_z(height,z,T,selected_levels=(0,1,2,3,4))
        t2 = smooth2d(t2,3, cenweight=4)
        return t2

    def forward_td2(self, input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            t2=self.forward_t2(input_samples,x,y,p,t)
            rh=self.forward_rh(input_samples,x,y,p,t)
            z=self.gather_variable('z',input_samples, x, y, p, t)

        height = self.altitude + 2
        rh2=self.interp_z(height, z, rh, selected_levels=(0, 1, 2, 3, 4))

        a=17.27
        b=237.7
        t=t2-273.15
        gamma=a*t/(b+t)+np.log(rh2/100+1e-16)

        td2=b*gamma/(a-gamma)+273.15

        return td2

    def forward_rh2(self,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            rh=self.forward_rh(input_samples,x,y,p,t,**kwargs)
            z = self.gather_variable('z', input_samples, x, y, p, t)

        height = self.altitude + 2
        rh2=self.interp_z(height, z, rh, selected_levels=(0, 1, 2, 3, 4))

        rh2 = smooth2d(rh2,3, cenweight=4)
        return rh2

    def forward_u10m(self, input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            z = self.gather_variable('z', input_samples, x, y, p, t)
            u = self.gather_variable('u', input_samples, x, y, p, t)

        height = self.altitude + 10
        data = self.interp_z(height, z, u, selected_levels=(0, 1, 2, 3, 4))
        data=smooth2d(data,3, cenweight=4)
        return data

    def forward_v10m(self, input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            z = self.gather_variable('z', input_samples, x, y, p, t)
            v = self.gather_variable('v', input_samples, x, y, p, t)

        data = self.interp_z(self.altitude + 10, z, v, selected_levels=(0, 1, 2, 3, 4))
        data=smooth2d(data,3, cenweight=4)

        return data

    def forward_u100m(self, input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            z = self.gather_variable('z', input_samples, x, y, p, t)
            u = self.gather_variable('u', input_samples, x, y, p, t)

        data = self.interp_z(self.altitude + 100, z, u, selected_levels=(0, 1, 2, 3, 4))
        data=smooth2d(data,3, cenweight=4)

        return data

    def forward_v100m(self, input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            z = self.gather_variable('z', input_samples, x, y, p, t)
            v = self.gather_variable('v', input_samples, x, y, p, t)

        data = self.interp_z(self.altitude + 100, z, v, selected_levels=(0, 1, 2, 3, 4))
        data=smooth2d(data,3, cenweight=4)

        return data

    def forward_wd10m(self,input_samples,x,y,p,t,**kwargs):
        with torch.no_grad():
            z = self.gather_variable('z', input_samples, x, y, p, t)
            u = self.gather_variable('u', input_samples, x, y, p, t)
            v = self.gather_variable('v', input_samples, x, y, p, t)
            w = self.gather_variable('w', input_samples, x, y, p, t)

        u = self.interp_z(self.altitude + 10, z, u, selected_levels=(0, 1, 2, 3, 4))
        v = self.interp_z(self.altitude + 10, z, v, selected_levels=(0, 1, 2, 3, 4))
        w = self.interp_z(self.altitude + 10, z, w, selected_levels=(0, 1, 2, 3, 4))
        wd=u**2+v**2+w**2
        data=wd**0.5
        data=smooth2d(data,3, cenweight=4)
        return data

    def forward_tp(self,input_samples, x, y, p, t, **kwargs):
        with torch.no_grad():
            T=self.gather_variable('T',input_samples,x,y,p,t)
            w=self.gather_variable('w',input_samples,x,y,p,t)

        press_data=np.array(self.press_levels)
        press_data=np.reshape(press_data,(-1,1,1))

    def interp_z(self,height,z,data,selected_levels=None):
        '''
        according z to interp variable
        :return:
        '''
        if selected_levels is None:
            selected_levels=list(range(0,len(z)))

        if not isinstance(selected_levels,list):
            selected_levels=list(selected_levels)

        z=z[selected_levels]
        L_r=np.expand_dims(z,axis=0)
        L_c=np.expand_dims(z,axis=1)
        idx=np.arange(0,z.shape[0])
        idx=idx.astype(np.int).tolist()
        L=(L_r-L_c)
        L[idx,idx,:,:]=1
        L_den=np.prod(L,axis=0)

        L_num=height-np.tile(L_c,[1,L_r.shape[1],1,1])
        L_num[idx,idx,:,:]=1
        L_num=np.prod(L_num,axis=0)

        L=L_num/L_den

        inter_data=L*data[[selected_levels]]
        inter_data=np.sum(inter_data,axis=0)
        return inter_data

    def gather_variable(self, var_name, input_samples, x, y, p, t):
        if var_name in self.intermidate_variables_dict.keys():
            return self.intermidate_variables_dict[var_name]
        with torch.no_grad():
            data = self.model.forward_single(var_name, input_samples)

        data = self.unpack_data(x, y, p, data, var_name)
        self.intermidate_variables_dict[var_name] = data

        return data

    def inverse_norm(self,data,press_level,norm_cfg):
        norm_type = norm_cfg['norm_type']
        norm_factor = norm_cfg['norm_factor']
        use_norm = norm_cfg['use_norm']
        if not use_norm:
            return data
        if norm_type.lower() == 'min_max':
            if len(norm_factor) == 2:
                data = data * (norm_factor[1][press_level] - norm_factor[0][press_level]) + norm_factor[0][press_level]
            else:
                data = data * (norm_factor[1][press_level] - norm_factor[0][press_level]) + norm_factor[0][press_level]
                data = data ** 2
                data = data + norm_factor[2][press_level]
        else:
            data = data * norm_factor[1][press_level] + norm_factor[0][press_level]

        return data

    def unpack_data(self,x:np.ndarray,y:np.ndarray,p:np.ndarray,data,var_name):

        if isinstance(data,torch.Tensor):
            data=data.detach().cpu().numpy()

        result_data=np.zeros([len(self.press_levels),self.lat_size,self.lon_size])
        for press_id,press_level in enumerate(self.press_levels):
            idx=np.nonzero(p==press_level)
            p_data=np.zeros([self.lat_size,self.lon_size])
            for i in idx[0]:
                y_id=y[i]
                x_id=x[i]
                d=data[i,0]
                p_data[y_id,x_id]=d


            p_data=self.inverse_norm(p_data,press_id,self.variable_dict[var_name])
            result_data[press_id]=p_data
        return result_data












