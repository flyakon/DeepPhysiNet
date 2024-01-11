import torch
import os
import shutil
import cv2

import numpy as np
from skimage import io
import argparse
import inspect
import pickle
import xarray

import matplotlib.pyplot as plt
import math
import torch
from torchvision.models import resnet18
from math import cos, pi


def adjust_learning_rate(optimizer, current_epoch,warmup_epoch,  lr_max=0.1):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class CoordUtils:
    def __init__(self,coord_file):
        with open(coord_file, 'rb') as fp:
            self.lon, self.lat = pickle.load(fp)
        self.lat_size,self.lon_size=self.lon.shape[0:2]
        coord_x = np.array(range(0, self.lon_size)) / (self.lon_size - 1)
        coord_y = np.array(range(0, self.lat_size)) / (self.lat_size - 1)
        self.lon_array=xarray.DataArray(self.lon[0,:],dims=['x'],coords=[coord_x.tolist()])
        self.lat_array = xarray.DataArray(self.lat[ :,0], dims=['y'], coords=[coord_y.tolist()])

        self.x_array=xarray.DataArray(coord_x*(self.lon_size - 1),dims=['lon'],coords=[self.lon[0,:]])
        self.y_array = xarray.DataArray(coord_y * (self.lat_size - 1), dims=['lat'], coords=[self.lat[:,0]])

    def geo2xy(self,ref_lon,ref_lat):
        if isinstance(ref_lon,torch.Tensor):
            ref_lon=ref_lon.detach().cpu().numpy()
        if isinstance(ref_lat, torch.Tensor):
            ref_lat = ref_lat.detach().cpu().numpy()
        x_array = self.x_array.interp(lon=ref_lon).data
        y_array = self.y_array.interp(lat=ref_lat).data
        return x_array,y_array

    def xy2geo(self,ref_x,ref_y,size_t):
        if isinstance(ref_x,torch.Tensor):
            ref_x=ref_x.detach().cpu().numpy()
        if isinstance(ref_y,torch.Tensor):
            ref_y=ref_y.detach().cpu().numpy()

        if not (isinstance(size_t,list) or isinstance(size_t,tuple)):
            size_t=(size_t,size_t)

        lat_size,lon_size=size_t
        lon_array=self.lon_array.interp(x=ref_x/(lon_size-1)).data
        lat_array=self.lat_array.interp(y=ref_y/(lat_size-1)).data

        return lon_array,lat_array

def get_variable_name(variable):
    print(locals())
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [f'{var_name}: {var_val}' for var_name, var_val in callers_local_vars if var_val is variable]


def update_single_params(arg_name,arg_value,config_cfg:dict,prefix=None):
    if prefix is None:
        prefix=get_variable_name(config_cfg)[0]

    for k,v in config_cfg.items():
        if k==arg_name:
            print('Find {0}, set it to {1}'.format(arg_name,arg_value))
            config_cfg[arg_name]=arg_value
        else:
            if isinstance(v,dict):
                update_single_params(arg_name,arg_value,v)
            else:
                continue

def update_param(args_dict:dict,config_cfgs,prefix=None):
    for k,v in args_dict.items():
        for config in config_cfgs:
            update_single_params(k,v,config,prefix)

def load_model(model_path,current_epoch=None,prefix='cub_model',rm_pre='module.'):

    '''
    载入模型,默认model文件夹中有一个latest.pth文件
    :param state_dict:
    :param model_path:
    :return:
    '''
    if os.path.isfile(model_path):
        model_file=model_path
    else:
        if current_epoch is None:
            model_file=os.path.join(model_path,'%s_latest.pth'%prefix)
        else:
            model_file = os.path.join(model_path, '%s_%d.pth'%(prefix,current_epoch))
    if not os.path.exists(model_file):
        print('warning:%s does not exist!'%model_file)
        return None,0,0
    print('start to resume from %s' % model_file)

    state_dict=torch.load(model_file)



    try:
        glob_step=state_dict.pop('gobal_step')
    except KeyError:
        print('warning:glob_step not in state_dict.')
        glob_step=0
    try:
        epoch=state_dict.pop('epoch')
    except KeyError:
        print('glob_step not in state_dict.')
        epoch=0
    if rm_pre is not None:
        rm_dict={}
        for k,v in state_dict.items():
            rm_dict[k.replace(rm_pre,'')]=v
        state_dict=rm_dict
    return state_dict,epoch+1,glob_step

def check_checkpoints(named_parameter,state_dict:dict):
    unpaired_count=0
    in_keys=[]
    for k,v in named_parameter:
        if k not in state_dict.keys():
            unpaired_count+=1
            print('model layer %s not in state_dict'%k)
        else:
            in_keys.append(k)
    for k in state_dict.keys():
        if k not in in_keys:
            print('%s not in model layer'%k)
    print('unpaired count %d'%unpaired_count)

def save_model(model,model_path,epoch,global_step,prefix='cub_model',max_keep=10):

    if isinstance(model,torch.nn.Module):
        state_dict=model.state_dict()
    else:
        state_dict=model
    state_dict['epoch']=epoch
    state_dict['gobal_step']=global_step

    model_file=os.path.join(model_path,'%s_%d.pth'%(prefix,epoch))
    torch.save(state_dict,model_file)
    shutil.copy(model_file,os.path.join(model_path,'%s_latest.pth'%prefix))

    # if epoch>max_keep:
    # 	for i in range(0,epoch-max_keep):
    # 		model_file=os.path.join(model_path,'%s_%d.pth'%(prefix,epoch))
    # 		if os.path.exists(model_file):
    # 			os.remove(model_file)



def calc_iou(prediction,gt):
    inter_xmin=np.maximum(prediction[:,0],gt[:,0])
    inter_ymin=np.maximum(prediction[:,1],gt[:,1])
    inter_xmax = np.minimum(prediction[:, 2], gt[:, 2])
    inter_ymax = np.minimum(prediction[:, 3], gt[:, 3])
    inter_height=np.maximum(inter_ymax-inter_ymin,0)
    inter_width=np.maximum(inter_xmax-inter_xmin,0)
    inter_area=inter_height*inter_width

    total_area=(prediction[:,3]-prediction[:,1])*(prediction[:,2]-prediction[:,0])+\
               (gt[:,3]-gt[:,1])*(gt[:,2]-gt[:,0])-inter_area
    iou=inter_area/total_area
    return iou

def rectangle(img,bndboxes,color):
    img_height, img_width, _ = img.shape
    img = cv2.rectangle(img, (int(bndboxes[0] * img_width), int(bndboxes[1] * img_height)),
                        (int(bndboxes[2] * img_width), int(bndboxes[3] * img_height)), color, 2)
    return img



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower in ('none','null','-1'):
        return None
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

osm_vis_dict={
        1:[255,0,0],
        2:[255,52,179],
        3:[0,0,205],
        4:[244,255,255],
        5:[0,100,0],
        6:[34,139,34],
        7:[173,255,47],
        8:[139,0,139],
        9:[139,123,139]
        }

glc_vis_dict={
        1:(249, 160, 255),
        2:(0, 99, 0),
        3:(99, 255, 0),
        4: (0, 255, 199),
        5:(0, 99, 255),
        6:(0, 0, 255),
        7:(99, 99, 51),
        8:(255, 0, 0),
        9:(191, 191, 191),
        10:(198, 239, 255)
        }

def vis_osm_result(img,result,result_path,file_name,label=None,fmask=None,img_size=256):
    img = img.cpu().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (img_size, img_size))

    img_height,img_width,_=img.shape
    result = result.detach().cpu().numpy()
    result = np.argmax(result, axis=0)+1
    result = cv2.resize(result, (img_size,img_size),interpolation=cv2.INTER_NEAREST)

    result_map=-np.ones([img_height,img_width,3])
    for k,v in osm_vis_dict.items():
        result_map=np.where(result[:,:,np.newaxis]==k,v,result_map)
    assert (result_map==-1).any() == False
    # result_map=result_map*fmask
    result_map = result_map.astype(np.uint8)
    if label is not None:
        label = label.detach().cpu().numpy()+1
        label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)
        label_map=-np.ones([img_height,img_width,3])
        for k,v in osm_vis_dict.items():
            label_map = np.where(label[:, :, np.newaxis] == k, v, label_map)
        assert (label_map == -1).any() == False
        if fmask is not None:
            fmask = fmask.cpu().numpy()
            fmask=cv2.resize(fmask,(img_size,img_size),interpolation=cv2.INTER_NEAREST)
            fmask = np.expand_dims(fmask, axis=-1)
            label_map=label_map*fmask

        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_osm_label.jpg'.format(file_name)), label_map,check_contrast=False)

    io.imsave(os.path.join(result_path, '{0}_osm_img.jpg'.format(file_name)), img,check_contrast=False)
    io.imsave(os.path.join(result_path, '{0}_osm_result.jpg'.format(file_name)), result_map,check_contrast=False)

def vis_glc_result(img,result,result_path,file_name,label=None,img_size=256):
    img = img.cpu().numpy()
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (img_size, img_size))

    img_height,img_width,_=img.shape
    result = result.detach().cpu().numpy()
    result = np.argmax(result, axis=0)+1
    result = cv2.resize(result, (img_size,img_size),interpolation=cv2.INTER_NEAREST)
    result_map=-np.ones([img_height,img_width,3])
    for k,v in glc_vis_dict.items():
        result_map=np.where(result[:,:,np.newaxis]==k,v,result_map)
    assert (result_map==-1).any() == False
    # result_map=result_map*fmask
    result_map = result_map.astype(np.uint8)
    if label is not None:
        label = label.detach().cpu().numpy()+1
        label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)

        label_map=-np.ones([img_height,img_width,3])
        for k,v in glc_vis_dict.items():
            label_map = np.where(label[:, :, np.newaxis] == k, v, label_map)
        assert (label_map == -1).any() == False
        label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_glc_label.jpg'.format(file_name)), label_map,check_contrast=False)

    io.imsave(os.path.join(result_path, '{0}_glc_img.jpg'.format(file_name)), img,check_contrast=False)
    result_map = cv2.resize(result_map, (img_size, img_size))
    io.imsave(os.path.join(result_path, '{0}_glc_result.jpg'.format(file_name)), result_map,check_contrast=False)


def vis_cropland_result(img,result,result_path,file_name,label=None,std=None,img_size=480):
    img = img.cpu().numpy()
    img=img[[2,1,0]]
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.resize(img, (img_size, img_size))

    img_height,img_width,_=img.shape
    result = result.detach().cpu().numpy()
    # result = cv2.resize(result, (img_size,img_size),interpolation=cv2.INTER_NEAREST)
    result_map=np.ones([img_height,img_width,3])
    result_map[:,:,1]=result[0]*255
    result_map=np.clip(result_map,0,255)

    # result_map=result_map*fmask
    result_map = result_map.astype(np.uint8)
    if label is not None:
        label = label.detach().cpu().numpy()
        # label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)

        label_map=np.zeros([img_height,img_width,3])
        label_map[:,:,1] = label[0]*255
        label_map=np.clip(label_map,0,255)
        label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_mean.jpg'.format(file_name)), label_map,check_contrast=False)

    if std is not None:
        std = std.detach().cpu().numpy()
        # label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)

        label_map=np.zeros([img_height,img_width,3])
        label_map[:,:,0] = std[0]*255

        label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_std.jpg'.format(file_name)), label_map,check_contrast=False)

    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(file_name)), img,check_contrast=False)
    result_map = cv2.resize(result_map, (img_size, img_size))
    io.imsave(os.path.join(result_path, '{0}_result.jpg'.format(file_name)), result_map,check_contrast=False)

def vis_downscale(result,result_path,file_name,label=None,norm_factor=(800,1100)):

    result = result.detach().cpu().numpy()
    result_map=cv2.applyColorMap()

    # result_map=result_map*fmask
    result_map = result_map.astype(np.uint8)
    if label is not None:
        label = label.detach().cpu().numpy()
        # label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)

        label_map=np.zeros([img_height,img_width,3])
        label_map[:,:,1] = label[0]*255
        label_map=np.clip(label_map,0,255)
        label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_mean.jpg'.format(file_name)), label_map,check_contrast=False)

    if std is not None:
        std = std.detach().cpu().numpy()
        # label = cv2.resize(label, (img_size, img_size),interpolation=cv2.INTER_NEAREST)

        label_map=np.zeros([img_height,img_width,3])
        label_map[:,:,0] = std[0]*255

        label_map = cv2.resize(label_map, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label_map = label_map.astype(np.uint8)

        io.imsave(os.path.join(result_path, '{0}_std.jpg'.format(file_name)), label_map,check_contrast=False)

    io.imsave(os.path.join(result_path, '{0}_img.jpg'.format(file_name)), img,check_contrast=False)
    result_map = cv2.resize(result_map, (img_size, img_size))
    io.imsave(os.path.join(result_path, '{0}_result.jpg'.format(file_name)), result_map,check_contrast=False)


if __name__=='__main__':
    pass