'''
@Project : deep-downscale 
@File    : interface_downscale.py
@Author  : Wenyuan Li
@Date    : 2022/2/11 19:09 
@Desc    :  
'''
import random
import time

import torch
import torch.nn as nn
import os
import numpy as np
import glob
import datetime
import tqdm

from DeepPhysiNet.utils import path_utils,gdal_utils
import shutil
from DeepPhysiNet.losses.builder import builder_loss
from DeepPhysiNet.utils.optims.builder import build_optim,build_lr_schedule
import torch.utils.data as data_utils

from DeepPhysiNet.metric.time_metric import TimeMetric
from torch.utils.tensorboard import SummaryWriter
from DeepPhysiNet.utils import downscale_utils,utils
from DeepPhysiNet.dataset import physics_dataset
from DeepPhysiNet.utils.position_encoding import SineCosPE
from DeepPhysiNet.model.physics_net import PhysicsNet

class InterfacePhysics(nn.Module):
    def __init__(self,meta_cfg:dict,net_cfg:dict,
                 obs_norm_cfg:dict,variable_cfg:dict,train_cfg:dict,test_cfg,inference_cfg:dict,**kwargs):
        super(InterfacePhysics, self).__init__()
        self.net_cfg=net_cfg
        self.obs_norm_cfg=obs_norm_cfg
        self.variable_cfg=variable_cfg
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        self.inference_cfg=inference_cfg
        self.physics_net=PhysicsNet(meta_cfg,net_cfg)
        img_size = self.train_cfg['img_size']
        self.pe=SineCosPE(3,include_input=False)
        if isinstance(img_size, int) or isinstance(img_size, float):
            self.lat_size, self.lon_size = img_size, img_size
        elif (isinstance(img_size, list) or isinstance(img_size, tuple)) and len(img_size) == 2:
            self.lat_size, self.lon_size = img_size
        else:
            raise NotImplementedError


    def save_model(self,checkpoint_path,epoch, global_step, prefix='physics',**kwargs):
        checkpoint_file=os.path.join(checkpoint_path,'%s_%d.pth'%(prefix,epoch))
        state_dict={}
        state_dict['model']=self.physics_net.state_dict()
        state_dict['epoch'] = epoch
        state_dict['gobal_step'] = global_step
        for k ,v in kwargs.items():
            state_dict[k]=v
        torch.save(state_dict, checkpoint_file)
        shutil.copy(checkpoint_file, os.path.join(checkpoint_path, '%s_latest.pth' % prefix))

    def load_model(self,checkpoint_path,current_epoch=None, prefix='downscale',map_location='cpu'):
        if os.path.isfile(checkpoint_path):
            model_file = checkpoint_path
        else:
            if current_epoch is None:
                model_file = os.path.join(checkpoint_path, '%s_latest.pth' % prefix)
            else:
                model_file = os.path.join(checkpoint_path, '%s_%d.pth' % (prefix, current_epoch))
        if not os.path.exists(model_file):
            print('warning:%s does not exist!' % model_file)
            return None, 0, 0
        print('start to resume from %s' % model_file)
        state_dict = torch.load(model_file,map_location=map_location)
        try:
            glob_step = state_dict.pop('gobal_step')
        except KeyError:
            print('warning:glob_step not in state_dict.')
            glob_step = 0
        try:
            epoch = state_dict.pop('epoch')
        except KeyError:
            print('glob_step not in state_dict.')
            epoch = 0

        return state_dict, epoch + 1, glob_step

    def gradient(self,y,x):
        grad= torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True,
                                   only_inputs=True,allow_unused=False)[0]
        # grad=grad.unsqueeze(dim=1)
        return grad

    def montion_equation_u(self, x, y, t, u, v, p, rio, f, loss, factor=1e-6):
        u_t = self.gradient(u, t)
        u_x = self.gradient(u, x)
        u_y = self.gradient(u, y)
        p_x = self.gradient(p, x)
        diff_term = u_t + u * u_x + v * u_y + p_x / rio
        const_term = f * v
        return loss(diff_term, const_term).float()*factor

    def montion_equation_v(self, x, y, t, u, v, p, rio, f, loss, factor=1e-6):

        v_t = self.gradient(v, t)
        v_x = self.gradient(v, x)
        v_y = self.gradient(v, y)
        p_y = self.gradient(p, y)
        diff_term = v_t + u * v_x + v * v_y + p_y / rio
        const_term = -f * u
        return loss(diff_term, const_term).float()*factor

    def continuous_equation(self, x, y, t, u, v, rio, loss, factor=1e-6):
        u_x = self.gradient(u, x)
        v_y = self.gradient(v, y)
        rio_t = self.gradient(rio, t)
        rio_x = self.gradient(rio, x)
        rio_y = self.gradient(rio, y)
        diff_term = rio_t + u * rio_x + v * rio_y + rio * u_x + rio * v_y
        const_term = torch.zeros_like(diff_term).float()
        return loss(diff_term, const_term).float()*factor

    def energy_equation(self, x, y, t, u, v, p, T, rio, q, loss, factor=1e-6, c_p=1005, L=2.5e6):
        T_t = self.gradient(T, t)
        T_x = self.gradient(T, x)
        T_y = self.gradient(T, y)

        p_t = self.gradient(p, t)
        p_x = self.gradient(p, x)
        p_y = self.gradient(p, y)

        q_t = self.gradient(q, t)
        q_x = self.gradient(q, x)
        q_y = self.gradient(q, y)

        T_term = c_p * (T_t + u * T_x + v * T_y)
        P_term = -(p_t + u * p_x + v * p_y) / (rio+1e-6)
        q_term = L * (q_t + u * q_x + v * q_y)
        diff_term = T_term + P_term + q_term
        const_term = torch.zeros_like(diff_term).float()
        return loss(diff_term, const_term).float()*factor

    def vapor_equation(self, x, y, t, u, v, p, T, q, loss, factor=1e-5, c_p=1005, L=2.5e6, R_v=461.5, R_d=287):
        def get_delta(p_t, q, q_s):
            cond = torch.logical_and(p_t < 0, torch.ge(q, q_s))
            return torch.where(cond, torch.ones_like(p_t), torch.zeros_like(p_t))

        def get_F(T, q, q_s):
            R = (1 + 0.608 * q) * R_d
            F = (L * R - c_p * R_v * T) / (c_p * R_v + T * T + L * L * q_s)
            F = F * q_s * T
            return F

        p_t = self.gradient(p, t)
        p_x = self.gradient(p, x)
        p_y = self.gradient(p, y)

        q_t = self.gradient(q, t)
        q_x = self.gradient(q, x)
        q_y = self.gradient(q, y)

        q_s = self.get_qs(p, T).detach()
        q_s=torch.maximum(q_s,torch.ones_like(q_s)*1e-6)
        delta = get_delta(p_t + u * p_x + v * p_y, q, q_s).detach()

        F = get_F(T, q, q_s).detach()

        P_term = -(p_t + u * p_x + v * p_y) * delta * F / (p+1e-6)
        q_term = q_t + u * q_x + v * q_y
        diff_term = P_term + q_term
        const_term = torch.zeros_like(diff_term).float()
        return loss(diff_term, const_term).float()*factor

    def gas_equation(self, p, T, rio, q, loss, factor=1e-5, R_d=287):
        terms = rio * (1 + 0.608 * q) * R_d * T
        return loss(p, terms).float()*factor

    def get_qs(self, p, T):
        t = T - 273.15
        e_s = 6.112 * torch.exp(17.67 * t / (t + 243.5)) * 100
        q_s = 0.622 * e_s / (p - 0.378 * e_s)
        return q_s

    def place_grid(self,n,dataset,pred_x_span,pred_y_span,pred_t_span,
                   margin_x=False,margin_y=False,margin_t=False):
        # time_len=self.time_len
        # # ref_pressure=[100000.,97500. ,95000. ,92500. ,90000. , 87500. , 85000. , 82500. , 80000.,
        # #             77500. , 75000. , 70000.,  65000.,  60000.,  55000.,  50000.,  45000.,  40000.,
        # #             35000.,  30000.]
        if margin_x:
            data=np.arange(0,self.lon_size,1)
            idx=np.random.randint(0,len(data),(n,))
            data=[float(data[i]) for i in idx]
            x=torch.tensor(data,dtype=torch.float).float()*self.dx
        else:
            x = torch.rand(n)*(pred_x_span-self.dx)
        if margin_y:
            data = np.arange(0, self.lat_size, 1)
            idx = np.random.randint(0, len(data), (n,))
            data = [float(data[i]) for i in idx]
            y = torch.tensor(data,dtype=torch.float).float()*self.dy
        else:
            y = torch.rand(n)*(pred_y_span-self.dy)


        if margin_t:
            data = np.arange(0, 2, 1)
            idx = np.random.randint(0, len(data), (n,))
            data = [float(data[i]) for i in idx]
            t=torch.tensor(data,dtype=torch.float).float()*self.dt
        else:
            # t = torch.rand(n)*(pred_t_span-self.dt)
            t = torch.rand(n)*pred_t_span

        f= dataset.get_coriolis(x/pred_x_span*self.lon_size,y/pred_y_span*self.lat_size)
        x=x.float()
        y=y.float()
        t=t.float()
        f=torch.from_numpy(f).float()
        x = torch.unsqueeze(x, dim=1)
        y = torch.unsqueeze(y, dim=1)
        t = torch.unsqueeze(t, dim=1)
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        return x,y,t,f


    def inverse_norm(self,u,v,P,T,q,rio,obs_norm_cfg,with_clip=False):

        def inverse_single(data,norm_cfg,with_clip=False):

            norm_type = norm_cfg['norm_type']
            norm_factor = norm_cfg['norm_factor']
            use_norm = norm_cfg['use_norm']
            bound=norm_cfg['bound']
            if not use_norm:
                return data
            if norm_type.lower() == 'min_max':

                if len(norm_factor) == 2:
                    data = data * (norm_factor[1] - norm_factor[0]) + norm_factor[0]
                else:
                    data = data * (norm_factor[1] - norm_factor[0]) + norm_factor[0]
                    data = data ** 2
                    data = data + norm_factor[2]
            else:
                data = data * norm_factor[1] + norm_factor[0]
            if with_clip:
                data = torch.clip(data, bound[0], bound[1])
            return data

        u = inverse_single(u,obs_norm_cfg['u10'])
        v = inverse_single(v, obs_norm_cfg['v10'])
        P = inverse_single(P, obs_norm_cfg['pres'],self.with_clip)
        T = inverse_single(T, obs_norm_cfg['t2'],self.with_clip)
        q = inverse_single(q, obs_norm_cfg['q2'],self.with_clip)
        rio = inverse_single(rio, obs_norm_cfg['rio'],self.with_clip)
        return u,v,P,T,q,rio


    def calc_rio(self,p,T,q,R_d=287):

        rio=(1+0.608*q)*R_d*T/p
        return rio.detach()


    def place_one_batch(self,x,y,t,f,field_data,input_data,forecast_h,criterion,loss_factor,
                        global_step,local_rank,device,summary=None,prefix='inter',log_step=100):
        f = f.to(device)
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        # inter_input = torch.cat((inter_x, inter_y, inter_t), dim=1)
        inter_input=self.encoding_coord(x, y, t, self.pred_t_span)

        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio = self.physics_net(field_data,inter_input,input_data,forecast_h)

        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio= self.inverse_norm(inter_u, inter_v,
                                                                                inter_P,inter_T, inter_q,inter_rio,
                                                                                obs_norm_cfg=self.obs_norm_cfg)
        montion_u_loss = self.montion_equation_u(x, y, t, inter_u, inter_v,
                                                 inter_P, inter_rio, f, criterion,
                                                 factor=loss_factor['motion_u_factor'])
        montion_v_loss = self.montion_equation_v(x, y, t, inter_u, inter_v,
                                                 inter_P, inter_rio, f, criterion,
                                                 factor=loss_factor['motion_v_factor'])
        continous_loss = self.continuous_equation(x, y, t, inter_u, inter_v, inter_rio, criterion,
                                                  factor=loss_factor['continuous_factor'])
        energy_loss = self.energy_equation(x, y, t, inter_u, inter_v, inter_P,
                                           inter_T, inter_rio, inter_q, criterion,
                                           factor=loss_factor['energy_factor'])
        vapor_loss = self.vapor_equation(x, y, t, inter_u, inter_v, inter_P, inter_T, inter_q, criterion,
                                         factor=loss_factor['vapor_factor'])
        gas_loss = self.gas_equation(inter_P, inter_T, inter_rio, inter_q, criterion,
                                     factor=loss_factor['gas_factor'])

        train_loss = montion_u_loss + montion_v_loss + energy_loss + continous_loss + vapor_loss + gas_loss

        if global_step % log_step == 1 and local_rank==0:
            summary.add_scalar('%s/total_loss' % prefix, train_loss, global_step)
            summary.add_scalar('%s/montion_u_loss' % prefix, montion_u_loss, global_step)
            summary.add_scalar('%s/montion_v_loss' % prefix, montion_v_loss, global_step)
            summary.add_scalar('%s/continous_loss' % prefix, continous_loss, global_step)
            summary.add_scalar('%s/energy_loss' % prefix, energy_loss, global_step)
            summary.add_scalar('%s/vapor_loss' % prefix, vapor_loss, global_step)
            summary.add_scalar('%s/gas_loss' % prefix, gas_loss, global_step)
            format_str = '%s:' % prefix
            format_str = format_str + '%s:%f,' % ('montion_u_loss', montion_u_loss.item())
            format_str = format_str + '%s:%f,' % ('montion_v_loss', montion_v_loss.item())
            format_str = format_str + '%s:%f,' % ('continous_loss', continous_loss.item())
            format_str = format_str + '%s:%f,' % ('energy_loss', energy_loss.item())
            format_str = format_str + '%s:%f,' % ('vapor_loss', vapor_loss.item())
            format_str = format_str + '%s:%f' % ('gas_loss', gas_loss.item())
            print(format_str)

        return train_loss.float()

    def encoding_coord(self,x,y,t,pred_t_span):
        #add PE
        x=x / self.dx / (self.lon_size-1)
        y=y / self.dy / (self.lat_size-1)
        t=t / pred_t_span
        if len(x.shape)==1:
            sample_input = torch.stack([x, y, t], dim=1)
        else:
            sample_input = torch.cat([x, y, t], dim=1)
        sample_input=self.pe.forward(sample_input)
        return sample_input

    def run_train_interface(self,**kwargs):
        batch_size = self.train_cfg['batch_size']
        device = self.train_cfg['device']
        num_epoch = self.train_cfg['num_epoch']
        num_workers = self.train_cfg['num_workers']
        self.dx = float(self.train_cfg['dx'])
        self.dy = float(self.train_cfg['dy'])
        time_step = self.train_cfg['lable_time_step']
        self.dt = float(60 * 60 * time_step)

        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path = kwargs['checkpoint_path']
        else:
            checkpoint_path = self.train_cfg['checkpoints']['checkpoints_path']

        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']

        vis_cfg=self.train_cfg['log']['vis_downscale_cfg']
        save_step = self.train_cfg['checkpoints']['save_step']
        log_step = self.train_cfg['log']['log_step']
        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']

        self.print_key_args(checkpoint_path=checkpoint_path,
                            log_path=log_path,
                            input_data_path=self.train_cfg['train_data']['input_path'],
                            device=device,
                            dx=self.dx,
                            dy=self.dy,
                            dt=self.dt)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        date_str=datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        log_file = os.path.join(log_path, 'log_%s.txt'%date_str)
        log_fp = open(log_file, 'w')

        if with_vis:
            if not os.path.exists(os.path.join(vis_path,'train_results')):
                os.makedirs(os.path.join(vis_path,'train_results'))
            if not os.path.exists(os.path.join(vis_path,'valid_results')):
                os.makedirs(os.path.join(vis_path,'valid_results'))
        self.physics_net.to(device)
        self.pe.to(device)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        pde_criterion = builder_loss(**self.train_cfg['losses']['pde_loss'])
        prediction_criterion = builder_loss(**self.train_cfg['losses']['prediction_loss'])
        variable_criterion=builder_loss(name='MSELoss')
        loss_factor=self.train_cfg['losses']['loss_factor']

        state_dict, current_epoch, global_step = self.load_model(checkpoint_path,prefix='physics')
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            log_fp.writelines('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.physics_net.load_state_dict(state_dict['model'],strict=True)
        optimizer = build_optim(params=[{'params':self.physics_net.parameters(),'initial_lr':self.train_cfg['optimizer']['lr']}],
                                **self.train_cfg['optimizer'])
        if 'lr_schedule' in self.train_cfg.keys():
            lr_schedule = build_lr_schedule(optimizer=optimizer, **self.train_cfg['lr_schedule'],last_epoch=current_epoch-1)
        train_dataset = physics_dataset.PhysicsDataset(**self.train_cfg['train_data'],
                                                       input_variable_cfg=self.variable_cfg,
                                                       out_variable_cfg=self.obs_norm_cfg,
                                                       dx=self.dx,dy=self.dy,

                                                       # input_time_step=time_step,
                                                       )
        self.train_cfg['valid_data']['auto_norm']=False
        valid_dataset = physics_dataset.PhysicsDataset(**self.train_cfg['valid_data'],
                                                       input_variable_cfg=self.variable_cfg,
                                                       out_variable_cfg=self.obs_norm_cfg,
                                                       dx=self.dx, dy=self.dy,
                                                       # input_time_step=time_step,
                                                       )
        pred_t_span=(train_dataset.input_time_step*train_dataset.input_time_step_nums)
        self.out_variable_cfg = train_dataset.out_variable_cfg

        pred_x_span=self.dx*self.lon_size
        pred_y_span = self.dy * self.lat_size
        pred_t_span = pred_t_span*60*60
        self.pred_t_span=pred_t_span
        train_dataloader = data_utils.DataLoader(train_dataset,batch_size, shuffle=True,
                                                          drop_last=True,
                                                 num_workers=num_workers,persistent_workers=False)
        valid_dataloader=data_utils.DataLoader(valid_dataset,batch_size, shuffle=True,
                                                          drop_last=True,
                                                 num_workers=0,persistent_workers=False)
        summary = SummaryWriter(log_path)
        time_metric = TimeMetric()
        self.physics_net.train()
        vis_utils=downscale_utils.VisUtils(**vis_cfg,img_size=(self.lon_size, self.lat_size))

        lr = optimizer.param_groups[0]['lr']
        summary.add_scalar('learning_rate', lr, global_step)
        print('set lr to:',lr)
        for epoch in range(current_epoch,num_epoch):
            valid_iter = iter(valid_dataloader)
            for batch_id,data in enumerate(train_dataloader):
                if global_step < 2000:
                    with_pde = False
                    self.with_clip = True
                else:
                    with_pde = True
                    self.with_clip = True

                global_step += 1
                field_data,margin_x,margin_y,margin_t,margin_data,margin_f,margin_input_data, \
                inter_x, inter_y, inter_t, inter_data, inter_f,forecast_h_unnorm,input_file=data

                field_data=field_data.to(device)
                forecast_h=forecast_h_unnorm/train_dataset.forecast_time_period
                forecast_h=forecast_h.float().to(device).unsqueeze(dim=0)
                margin_x = margin_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_y = margin_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_t = margin_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_f = margin_f.float().to(device).squeeze(dim=0)
                margin_data = margin_data.float().to(device).squeeze(dim=0)
                margin_input_data = margin_input_data.float().to(device).squeeze(dim=0)

                inter_x = inter_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_y = inter_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_t = inter_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_f = inter_f[0].float().to(device).squeeze(dim=0)
                inter_data = inter_data.float().to(device).squeeze(dim=0)
                # inter_factor = inter_factor.float().to(device).squeeze(dim=0)

                margin_input = self.encoding_coord(margin_x, margin_y, margin_t,
                                                   pred_t_span)

                margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = self.physics_net.forward(field_data,
                                                                                                        margin_input,
                                                                                                        margin_input_data,
                                                                                                        forecast_h)
                margin_loss = prediction_criterion(
                    torch.cat((margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio), dim=1),
                    margin_data).float()
                margin_loss = margin_loss * loss_factor['margin_factor']
                loss_dict={'margin_loss':margin_loss}
                if with_pde:
                    # x,y,t,f=self.place_grid(batch_size_inter,train_dataset,pred_x_span,pred_y_span,pred_t_span,
                    #                         margin_t=False)
                    x,y,t,f=inter_x,inter_y,inter_t,inter_f
                    # x = torch.unsqueeze(x, dim=1)
                    # y = torch.unsqueeze(y, dim=1)
                    # t = torch.unsqueeze(t, dim=1)
                    x = x.requires_grad_(True)
                    y = y.requires_grad_(True)
                    t = t.requires_grad_(True)
                    inter_loss=self.place_one_batch(x,y,t,f,field_data,inter_data,forecast_h,pde_criterion,loss_factor,summary,
                                                    global_step,device,'inter',log_step=log_step)

                    margin_x = margin_x.requires_grad_(True)
                    margin_y = margin_y.requires_grad_(True)
                    margin_t = margin_t.requires_grad_(True)

                    margin_pde_loss = self.place_one_batch(margin_x, margin_y, margin_t, margin_f, field_data,
                                                           margin_input_data,forecast_h,
                                                           pde_criterion, loss_factor, summary, global_step,
                                                           device, 'margin',log_step=log_step)
                    loss_dict['inter_pde_loss']=inter_loss
                    loss_dict['margin_pde_loss'] = margin_pde_loss
                    # loss_dict['margin_pde_loss'] = margin_pde_loss

                train_loss=0
                for key,v in loss_dict.items():
                    train_loss=train_loss+v

                optimizer.zero_grad()
                train_loss.backward()

                if global_step%log_step==1:
                    total_sum = 0
                    for key,p in self.physics_net.named_parameters():
                        param_norm = p.grad.detach().norm(2).item()
                        total_sum=total_sum+param_norm

                torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(),max_norm=2.5e7)
                optimizer.step()

                if global_step % log_step == 1:
                    self.physics_net.eval()

                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio=\
                        self.inverse_norm(margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio,self.obs_norm_cfg)
                    label_u,label_v,label_p,label_T,label_q,label_rio=\
                        self.inverse_norm(margin_data[:,0:1],margin_data[:,1:2],margin_data[:,2:3],margin_data[:,3:4],margin_data[:,4:5],
                                          margin_data[:,5:6],self.obs_norm_cfg)
                    margin_u_loss=variable_criterion(margin_u,label_u).detach()
                    margin_v_loss = variable_criterion(margin_v, label_v).detach()
                    margin_p_loss = variable_criterion(margin_p, label_p).detach()
                    margin_T_loss = variable_criterion(margin_T, label_T).detach()
                    margin_q_loss = variable_criterion(margin_q, label_q).detach()
                    margin_rio_loss = variable_criterion(margin_rio, label_rio).detach()

                    if with_vis:
                        self.with_clip=False
                        if with_pde:
                            time_id=np.random.randint(0,train_dataset.input_time_step*train_dataset.input_time_step_nums+1,(1,))[0]
                        else:
                            time_id=0
                        x_list = []
                        y_list = []
                        t_list = []
                        for x in range(0, self.lon_size):
                            for y in range(0, self.lat_size):
                                x_list.append(x)
                                y_list.append(y)
                                t_list.append(time_id)

                        x_list,y_list,t_list,test_data,\
                        margin_f=train_dataset.get_margin_grid(input_file[0],
                                                                                    x_list,y_list,t_list)
                        x_list=x_list.float().to(device)
                        y_list=y_list.float().to(device)
                        t_list=t_list.float().to(device)
                        test_data=test_data.float().to(device).squeeze(dim=0)
                        # t_list = torch.tensor(t_list) / 100
                        # input = torch.stack([x_list, y_list, t_list], dim=1).to(device)
                        input=self.encoding_coord(x_list,y_list,t_list,pred_t_span)
                        with torch.no_grad():
                            inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio = self.physics_net(field_data,input,
                                                                                                    test_data,
                                                                                                    forecast_h)
                        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio= \
                            self.inverse_norm(inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio,
                                              obs_norm_cfg=self.obs_norm_cfg)
                        # inter_p = inter_p * 255
                        result_v = np.zeros([self.lat_size, self.lon_size])
                        result_u = np.zeros([self.lat_size, self.lon_size])
                        result_P = np.zeros([self.lat_size, self.lon_size])
                        result_T = np.zeros([self.lat_size, self.lon_size])
                        result_q = np.zeros([self.lat_size, self.lon_size])
                        result_rio = np.zeros([self.lat_size, self.lon_size])

                        x_list=x_list/self.dx
                        y_list=y_list/self.dy
                        y_list = y_list.cpu().numpy()
                        x_list = x_list.cpu().numpy()
                        inter_P = inter_P.detach().cpu().numpy()
                        inter_u = inter_u.detach().cpu().numpy()
                        inter_v = inter_v.detach().cpu().numpy()
                        inter_T = inter_T.detach().cpu().numpy()
                        inter_q = inter_q.detach().cpu().numpy()
                        inter_rio = inter_rio.detach().cpu().numpy()

                        for id, (x, y) in enumerate(zip(x_list, y_list)):
                            y = int(y)
                            x = int(x)
                            result_u[y, x] = inter_u[id]
                            result_v[y, x] = inter_v[id]
                            result_P[y, x] = inter_P[id]
                            result_T[y, x] = inter_T[id]
                            result_q[y,x] = inter_q[id]
                            result_rio[y,x] = inter_rio[id]
                        forecast_h_unnorm=forecast_h_unnorm.item()
                        # forecast_h=np.round(forecast_h,4)*train_dataset.forecast_time_period
                        # forecast_h=forecast_h*train_dataset.forecast_time_period
                        result_file = os.path.join(vis_path,'train_results', '%d_result_u_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_u,result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_v_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_v, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_P_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_P, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_T_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_T, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_q_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_q, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_rio_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_rio, result_file)
                    fps = time_metric.get_fps(log_step * batch_size)
                    time_metric.reset()
                    print('=============================training==============================')
                    format_str = 'epoch:%d/%d,batch:%d/%d,iter:%d/%d,' % (
                        epoch, num_epoch, batch_id, len(train_dataloader), global_step,
                        len(train_dataloader) * num_epoch)
                    format_str = format_str + '%s:%f,' % ('train loss', train_loss)
                    for k, v in loss_dict.items():
                        format_str = format_str + '%s:%f,' % (k, v.item())
                    format_str = format_str + '%s:%03dh,' % ('forecast', forecast_h_unnorm)
                    format_str = format_str + '%s:%f,%s:%f' % ('grad',total_sum,'fps', fps)
                    print(format_str)
                    log_fp.writelines('%s\n' % format_str)
                    log_fp.flush()
                    summary.add_scalar('training/total_loss', train_loss, global_step)
                    for k, v in loss_dict.items():
                        summary.add_scalar('training/%s' % k, v.detach(), global_step)
                    summary.add_scalar('training/margin_u_loss',margin_u_loss.detach(),global_step)
                    summary.add_scalar('training/margin_v_loss', margin_v_loss.detach(), global_step)
                    summary.add_scalar('training/margin_p_loss', margin_p_loss.detach(), global_step)
                    summary.add_scalar('training/margin_T_loss', margin_T_loss.detach(), global_step)
                    summary.add_scalar('training/margin_q_loss', margin_q_loss.detach(), global_step)
                    summary.add_scalar('training/margin_rio_loss', margin_rio_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_u_loss' % forecast_h, margin_u_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_v_loss' % forecast_h, margin_v_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_p_loss' % forecast_h, margin_p_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_T_loss' % forecast_h, margin_T_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_q_loss' % forecast_h, margin_q_loss.detach(), global_step)
                    summary.add_scalar('training_f%03d/margin_rio_loss' % forecast_h, margin_rio_loss.detach(), global_step)
                    print('=============================end==============================')
                    ###############validation
                    self.with_clip = True
                    try:
                        data=valid_iter.__next__()
                    except:
                        valid_iter=iter(valid_dataloader)
                        data = valid_iter.__next__()
                    field_data, margin_x, margin_y, margin_t, margin_data, margin_f,margin_input_data, \
                    inter_x, inter_y, inter_t, inter_data, inter_f, forecast_h_unnorm, input_file = data

                    field_data = field_data.to(device)
                    forecast_h=forecast_h_unnorm/valid_dataset.forecast_time_period
                    forecast_h = forecast_h.float().to(device).unsqueeze(dim=0)
                    margin_x = margin_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_y = margin_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_t = margin_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_f = margin_f.float().to(device).squeeze(dim=0)
                    margin_data = margin_data.float().to(device).squeeze(dim=0)
                    margin_input_data = margin_input_data.float().to(device).squeeze(dim=0)

                    inter_x = inter_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_y = inter_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_t = inter_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_f = inter_f[0].float().to(device).squeeze(dim=0)
                    inter_data = inter_data.float().to(device).squeeze(dim=0)

                    margin_input = self.encoding_coord(margin_x, margin_y, margin_t,
                                                       pred_t_span)
                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = self.physics_net.forward(field_data,
                                                                                                            margin_input,
                                                                                                            margin_input_data,
                                                                                                            forecast_h)
                    margin_loss = prediction_criterion(
                        torch.cat((margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio), dim=1),
                        margin_data).float()
                    margin_loss = margin_loss * loss_factor['margin_factor']
                    loss_dict = {'margin_loss': margin_loss}
                    if with_pde:
                        # x, y, t, f = self.place_grid(batch_size_inter, valid_dataset, pred_x_span, pred_y_span,
                        #                              pred_t_span,
                        #                              margin_t=False)
                        x=inter_x
                        y=inter_y
                        t=inter_t
                        f=inter_f
                        # x = torch.unsqueeze(x, dim=1)
                        # y = torch.unsqueeze(y, dim=1)
                        # t = torch.unsqueeze(t, dim=1)
                        x = x.requires_grad_(True)
                        y = y.requires_grad_(True)
                        t = t.requires_grad_(True)
                        inter_loss = self.place_one_batch(x, y, t, f, field_data,inter_data, forecast_h, pde_criterion,
                                                          loss_factor, summary, global_step, device, 'inter',log_step=log_step).detach()

                        margin_x = margin_x.requires_grad_(True)
                        margin_y = margin_y.requires_grad_(True)
                        margin_t = margin_t.requires_grad_(True)
                        #
                        margin_pde_loss = self.place_one_batch(margin_x, margin_y, margin_t, margin_f, field_data,
                                                               margin_input_data,forecast_h,
                                                               pde_criterion, loss_factor, summary, global_step,
                                                               device, 'margin',log_step=log_step).detach()
                        loss_dict['inter_pde_loss'] = inter_loss.detach()
                        loss_dict['margin_pde_loss'] = margin_pde_loss.detach()

                    valid_loss = 0
                    for key, v in loss_dict.items():
                        valid_loss = valid_loss + v.detach()

                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = \
                        self.inverse_norm(margin_u.detach(), margin_v.detach(), margin_p.detach(), margin_T.detach(),
                                          margin_q.detach(), margin_rio.detach(),
                                          self.obs_norm_cfg)
                    label_u, label_v, label_p, label_T, label_q, label_rio = \
                        self.inverse_norm(margin_data[:, 0:1], margin_data[:, 1:2], margin_data[:, 2:3],
                                          margin_data[:, 3:4], margin_data[:, 4:5],
                                          margin_data[:, 5:6], self.obs_norm_cfg)
                    margin_u_loss = variable_criterion(margin_u, label_u).detach()
                    margin_v_loss = variable_criterion(margin_v, label_v).detach()
                    margin_p_loss = variable_criterion(margin_p, label_p).detach()
                    margin_T_loss = variable_criterion(margin_T, label_T).detach()
                    margin_q_loss = variable_criterion(margin_q, label_q).detach()
                    margin_rio_loss = variable_criterion(margin_rio, label_rio).detach()

                    print('=============================validation==============================')
                    ref_h = forecast_h_unnorm.item()
                    # ref_h=np.round(ref_h,2)*train_dataset.forecast_time_period
                    # ref_h = ref_h * train_dataset.forecast_time_period
                    format_str = 'epoch:%d/%d,batch:%d/%d,iter:%d/%d,' % (
                        epoch, num_epoch, batch_id, len(train_dataloader), global_step,
                        len(train_dataloader) * num_epoch)
                    format_str = format_str + '%s:%f,' % ('valid loss', valid_loss)
                    for k, v in loss_dict.items():
                        format_str = format_str + '%s:%f,' % (k, v.item())
                    format_str = format_str + '%s:%03dh,' % ('forecast', ref_h)
                    format_str = format_str + '%s:%f' % ('fps', fps)
                    print(format_str)
                    log_fp.writelines('%s\n' % format_str)
                    log_fp.flush()
                    summary.add_scalar('validation/total_loss', valid_loss, global_step)
                    for k, v in loss_dict.items():
                        summary.add_scalar('validation/%s' % k, v, global_step)

                    summary.add_scalar('validation/margin_u_loss', margin_u_loss.detach(), global_step)
                    summary.add_scalar('validation/margin_v_loss', margin_v_loss.detach(), global_step)
                    summary.add_scalar('validation/margin_p_loss', margin_p_loss.detach(), global_step)
                    summary.add_scalar('validation/margin_T_loss', margin_T_loss.detach(), global_step)
                    summary.add_scalar('validation/margin_q_loss', margin_q_loss.detach(), global_step)
                    summary.add_scalar('validation/margin_rio_loss', margin_rio_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_u_loss'%ref_h, margin_u_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_v_loss'%ref_h, margin_v_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_p_loss'%ref_h, margin_p_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_T_loss'%ref_h, margin_T_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_q_loss'%ref_h, margin_q_loss.detach(), global_step)
                    summary.add_scalar('validation_f%03d/margin_rio_loss'%ref_h, margin_rio_loss.detach(), global_step)
                    print('=============================end==============================')

                    if with_vis:
                        self.with_clip=False
                        if with_pde:
                            time_id = \
                            np.random.randint(0, valid_dataset.input_time_step * valid_dataset.input_time_step_nums + 1,
                                              (1,))[0]
                        else:
                            time_id=0
                        x_list = []
                        y_list = []
                        t_list = []
                        for x in range(0, self.lon_size):
                            for y in range(0, self.lat_size):
                                x_list.append(x)
                                y_list.append(y)
                                t_list.append(time_id)
                        x_list, y_list, t_list, test_data, \
                        margin_f = valid_dataset.get_margin_grid(input_file[0],
                                                                                x_list, y_list, t_list) #* self.time_len
                        x_list=x_list.float().to(device)
                        y_list=y_list.float().to(device)
                        t_list=t_list.float().to(device)
                        test_data=test_data.float().to(device).squeeze(dim=0)
                        # t_list = torch.tensor(t_list) / 100
                        # input = torch.stack([x_list, y_list, t_list], dim=1).to(device)
                        input=self.encoding_coord(x_list,y_list,t_list,pred_t_span)
                        with torch.no_grad():
                            inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio = self.physics_net(field_data,input,
                                                                                                    test_data,forecast_h)
                        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio= \
                            self.inverse_norm(inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio,
                                              obs_norm_cfg=self.obs_norm_cfg)
                        # inter_p = inter_p * 255
                        result_v = np.zeros([self.lat_size, self.lon_size])
                        result_u = np.zeros([self.lat_size, self.lon_size])
                        result_P = np.zeros([self.lat_size, self.lon_size])
                        result_T = np.zeros([self.lat_size, self.lon_size])
                        result_q = np.zeros([self.lat_size, self.lon_size])
                        result_rio = np.zeros([self.lat_size, self.lon_size])

                        x_list=x_list/self.dx
                        y_list=y_list/self.dy
                        y_list = y_list.cpu().numpy()
                        x_list = x_list.cpu().numpy()
                        inter_P = inter_P.detach().cpu().numpy()
                        inter_u = inter_u.detach().cpu().numpy()
                        inter_v = inter_v.detach().cpu().numpy()
                        inter_T = inter_T.detach().cpu().numpy()
                        inter_q = inter_q.detach().cpu().numpy()
                        inter_rio = inter_rio.detach().cpu().numpy()

                        for id, (x, y) in enumerate(zip(x_list, y_list)):
                            y = int(y)
                            x = int(x)
                            result_u[y, x] = inter_u[id]
                            result_v[y, x] = inter_v[id]
                            result_P[y, x] = inter_P[id]
                            result_T[y, x] = inter_T[id]
                            result_q[y,x] = inter_q[id]
                            result_rio[y,x] = inter_rio[id]
                        forecast_h=forecast_h.item()
                        forecast_h=np.round(forecast_h,4)*train_dataset.forecast_time_period
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_u_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_u,result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_v_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_v, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_P_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_P, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_T_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_T, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_q_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_q, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_rio_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_rio, result_file)

                    self.physics_net.train()

            if epoch % save_step==0:
                if 'lr_schedule' in self.train_cfg.keys():
                    lr_schedule.step()
                    summary.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'],
                                       global_step)
                lr = optimizer.param_groups[0]['lr']
                summary.add_scalar('learning_rate', lr, global_step)
                self.save_model(checkpoint_path, epoch, global_step,  prefix='physics',
                                dx=self.dx,dy=self.dy,dt=self.dt,pred_x_span=pred_x_span,
                                pred_y_span=pred_y_span,pred_t_span=pred_t_span,
                                label_time_step=time_step,input_variable_cfg=train_dataset.input_variable_cfg,
                                input_time_step=train_dataset.input_time_step,
                                input_time_step_nums=train_dataset.input_time_step_nums,
                                obs_norm_cfg=self.obs_norm_cfg,
                                start_time=train_dataset.start_time,end_time=train_dataset.end_time)


    def run_train_interface_dist(self,**kwargs):
        batch_size = self.train_cfg['batch_size']
        # device = self.train_cfg['device']
        num_epoch = self.train_cfg['num_epoch']
        num_workers = self.train_cfg['num_workers']
        self.dx = float(self.train_cfg['dx'])
        self.dy = float(self.train_cfg['dy'])
        time_step = self.train_cfg['lable_time_step']
        self.dt = float(60 * 60 * time_step)
        with_pde=self.train_cfg['with_pde']
        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path = kwargs['checkpoint_path']
        else:
            checkpoint_path = self.train_cfg['checkpoints']['checkpoints_path']

        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']

        vis_cfg=self.train_cfg['log']['vis_downscale_cfg']
        save_step = self.train_cfg['checkpoints']['save_step']
        log_step = self.train_cfg['log']['log_step']
        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']

        local_rank = torch.distributed.get_rank()
        # world_size = torch.distributed.world_size()

        if local_rank==0:
            self.print_key_args(checkpoint_path=checkpoint_path,
                                log_path=log_path,
                                input_data_path=self.train_cfg['train_data']['input_path'],
                                # device=device,
                                dx=self.dx,
                                dy=self.dy,
                                dt=self.dt)

            if not os.path.exists(log_path):
                os.makedirs(log_path)

            date_str=datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            log_file = os.path.join(log_path, 'log_%s.txt'%date_str)
            log_fp = open(log_file, 'w')

            if with_vis:
                if not os.path.exists(os.path.join(vis_path,'train_results')):
                    os.makedirs(os.path.join(vis_path,'train_results'))
                if not os.path.exists(os.path.join(vis_path,'valid_results')):
                    os.makedirs(os.path.join(vis_path,'valid_results'))

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        self.physics_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.physics_net).to(device)
        # self.physics_net.to(device)
        self.physics_net=torch.nn.parallel.DistributedDataParallel(self.physics_net,
                                                  # device_ids=[local_rank],
                                                  # output_device=local_rank,
                                                                   broadcast_buffers=False,
                                                                   find_unused_parameters=False)
        self.pe.to(device)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        pde_criterion = builder_loss(**self.train_cfg['losses']['pde_loss'])
        prediction_criterion = builder_loss(**self.train_cfg['losses']['prediction_loss'])
        variable_criterion=builder_loss(name='MSELoss')
        loss_factor=self.train_cfg['losses']['loss_factor']

        state_dict, current_epoch, global_step = self.load_model(checkpoint_path,prefix='physics',
                                                                 map_location='cuda:{}'.format(local_rank))
        if state_dict is not None:
            if local_rank==0:
                print('resume from epoch %d global_step %d' % (current_epoch, global_step))
                log_fp.writelines('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.physics_net.load_state_dict(state_dict['model'],strict=True)
            # torch.distributed.barrier()
        optimizer = build_optim(params=[{'params':self.physics_net.parameters(),
                                         'initial_lr':self.train_cfg['optimizer']['lr']}],
                                **self.train_cfg['optimizer'])
        if 'lr_schedule' in self.train_cfg.keys():
            lr_schedule = build_lr_schedule(optimizer=optimizer, **self.train_cfg['lr_schedule'],
                                            last_epoch=current_epoch-1)
        train_dataset = physics_dataset.PhysicsDataset(**self.train_cfg['train_data'],
                                                       input_variable_cfg=self.variable_cfg,
                                                       out_variable_cfg=self.obs_norm_cfg,
                                                       dx=self.dx,dy=self.dy,local_rank=local_rank
                                                       # input_time_step=time_step,
                                                       )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # if local_rank==0:
        self.train_cfg['valid_data']['auto_norm']=False
        valid_dataset = physics_dataset.PhysicsDataset(**self.train_cfg['valid_data'],
                                                       input_variable_cfg=self.variable_cfg,
                                                       out_variable_cfg=self.obs_norm_cfg,
                                                       dx=self.dx, dy=self.dy,
                                                       # input_time_step=time_step,
                                                       local_rank=local_rank
                                                       )
        # valid_sample=torch.utils.data.distributed.DistributedSampler(valid_dataset)
        pred_t_span=(train_dataset.input_time_step*train_dataset.input_time_step_nums)
        self.out_variable_cfg = train_dataset.out_variable_cfg

        pred_x_span=self.dx*self.lon_size
        pred_y_span = self.dy * self.lat_size
        pred_t_span = pred_t_span*60*60
        self.pred_t_span=pred_t_span
        train_dataloader = data_utils.DataLoader(train_dataset,batch_size,
                                                          drop_last=True,sampler=train_sampler,
                                                 num_workers=num_workers,persistent_workers=True,
                                                 pin_memory=True)
        # if local_rank==0:
        valid_dataloader=data_utils.DataLoader(valid_dataset,batch_size,shuffle=True,
                                                          drop_last=True,
                                                 num_workers=1,persistent_workers=True,pin_memory=True)
        if local_rank == 0:
            summary = SummaryWriter(log_path)

            lr = optimizer.param_groups[0]['lr']
            vis_utils = downscale_utils.VisUtils(**vis_cfg, img_size=(self.lon_size, self.lat_size))
            summary.add_scalar('learning_rate', lr, global_step)
            print('set lr to:', lr)
        else:
            summary=None
        self.physics_net.train()
        time_metric = TimeMetric()
        for epoch in range(current_epoch,num_epoch):
            # if local_rank==0:
            valid_iter = iter(valid_dataloader)
            if local_rank==0:
                dataloader=tqdm.tqdm(train_dataloader)
            else:
                dataloader=train_dataloader
            for batch_id,data in enumerate(dataloader):
                # if local_rank == 0: print('read')
                # torch.distributed.barrier()
                if global_step < 2000:
                    with_pde = False
                    self.with_clip = True
                else:
                    with_pde = with_pde
                    self.with_clip = True
                # if local_rank==0:print(global_step)
                global_step += 1

                field_data,margin_x,margin_y,margin_t,margin_data,margin_f,margin_input_data, \
                inter_x, inter_y, inter_t, inter_data, inter_f,forecast_h_unnorm,input_file=data

                field_data=field_data.to(device)
                tmp_time_step=train_dataset.input_time_step*train_dataset.input_time_step_nums
                forecast_h = forecast_h_unnorm//tmp_time_step*tmp_time_step/train_dataset.forecast_time_period
                forecast_h=forecast_h.float().to(device).unsqueeze(dim=0)
                margin_x = margin_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_y = margin_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_t = margin_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                margin_f = margin_f.float().to(device).squeeze(dim=0)
                margin_data = margin_data.float().to(device).squeeze(dim=0)
                margin_input_data = margin_input_data.float().to(device).squeeze(dim=0)

                inter_x = inter_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_y = inter_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_t = inter_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                inter_f = inter_f[0].float().to(device).squeeze(dim=0)
                inter_data = inter_data.float().to(device).squeeze(dim=0)

                margin_input = self.encoding_coord(margin_x, margin_y, margin_t,
                                                   pred_t_span)
                # if local_rank==0:print('data')

                margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = self.physics_net.forward(field_data,
                                                                                                        margin_input,
                                                                                                        margin_input_data,
                                                                                                        forecast_h)
                # if local_rank==0:print('model')

                margin_loss = prediction_criterion(
                    torch.cat((margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio), dim=1),
                    margin_data).float()
                margin_loss = margin_loss * loss_factor['margin_factor']
                loss_dict={'margin_loss':margin_loss}
                if with_pde:

                    x,y,t,f=inter_x,inter_y,inter_t,inter_f
                    x = x.requires_grad_(True)
                    y = y.requires_grad_(True)
                    t = t.requires_grad_(True)
                    inter_loss=self.place_one_batch(x,y,t,f,field_data,inter_data,forecast_h,pde_criterion,
                                                    loss_factor,
                                                    global_step,local_rank,device,summary,'inter',log_step=log_step)

                    margin_x = margin_x.requires_grad_(True)
                    margin_y = margin_y.requires_grad_(True)
                    margin_t = margin_t.requires_grad_(True)

                    margin_pde_loss = self.place_one_batch(margin_x, margin_y, margin_t, margin_f, field_data,
                                                           margin_input_data,forecast_h,
                                                           pde_criterion, loss_factor,
                                                           global_step,local_rank,
                                                           device,summary, 'margin',log_step=log_step)
                    loss_dict['inter_pde_loss']=inter_loss
                    loss_dict['margin_pde_loss'] = margin_pde_loss

                # if local_rank==0:print('loss')
                train_loss=0
                for key,v in loss_dict.items():
                    train_loss=train_loss+v

                optimizer.zero_grad()
                # if local_rank==0:print('backward')
                train_loss.backward()

                if global_step%log_step==1 :
                    total_sum = 0
                    for key,p in self.physics_net.named_parameters():
                        param_norm = p.grad.detach().norm(2).item()
                        total_sum=total_sum+param_norm

                torch.nn.utils.clip_grad_norm_(self.physics_net.parameters(),max_norm=2.5e7)
                optimizer.step()
                # if local_rank==0:print('step')
                if global_step % log_step == 1 :
                    self.physics_net.eval()

                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio=\
                        self.inverse_norm(margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio,self.obs_norm_cfg)
                    label_u,label_v,label_p,label_T,label_q,label_rio=\
                        self.inverse_norm(margin_data[:,0:1],margin_data[:,1:2],margin_data[:,2:3],margin_data[:,3:4],margin_data[:,4:5],
                                          margin_data[:,5:6],self.obs_norm_cfg)
                    margin_u_loss=variable_criterion(margin_u,label_u).detach()
                    margin_v_loss = variable_criterion(margin_v, label_v).detach()
                    margin_p_loss = variable_criterion(margin_p, label_p).detach()
                    margin_T_loss = variable_criterion(margin_T, label_T).detach()
                    margin_q_loss = variable_criterion(margin_q, label_q).detach()
                    margin_rio_loss = variable_criterion(margin_rio, label_rio).detach()
                    forecast_h_unnorm = forecast_h_unnorm.item()
                    forecast_idx = forecast_h.item() * train_dataset.forecast_time_period
                    if with_vis and local_rank==0:
                        self.with_clip=False
                        if with_pde:
                            time_id=np.random.randint(0,train_dataset.input_time_step*train_dataset.input_time_step_nums+1,(1,))[0]
                        else:
                            time_id=0
                        x_list = []
                        y_list = []
                        t_list = []
                        for x in range(0, self.lon_size):
                            for y in range(0, self.lat_size):
                                x_list.append(x)
                                y_list.append(y)
                                t_list.append(time_id)

                        x_list,y_list,t_list,test_data,\
                        margin_f=train_dataset.get_margin_grid(input_file[0],
                                                                                    x_list,y_list,t_list)
                        x_list=x_list.float().to(device)
                        y_list=y_list.float().to(device)
                        t_list=t_list.float().to(device)
                        test_data=test_data.float().to(device).squeeze(dim=0)
                        # t_list = torch.tensor(t_list) / 100
                        # input = torch.stack([x_list, y_list, t_list], dim=1).to(device)
                        input=self.encoding_coord(x_list,y_list,t_list,pred_t_span)
                        with torch.no_grad():
                            inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio = self.physics_net(field_data,input,
                                                                                                    test_data,
                                                                                                    forecast_h)
                        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio= \
                            self.inverse_norm(inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio,
                                              obs_norm_cfg=self.obs_norm_cfg)
                        # inter_p = inter_p * 255
                        result_v = np.zeros([self.lat_size, self.lon_size])
                        result_u = np.zeros([self.lat_size, self.lon_size])
                        result_P = np.zeros([self.lat_size, self.lon_size])
                        result_T = np.zeros([self.lat_size, self.lon_size])
                        result_q = np.zeros([self.lat_size, self.lon_size])
                        result_rio = np.zeros([self.lat_size, self.lon_size])

                        x_list=x_list/self.dx
                        y_list=y_list/self.dy
                        y_list = y_list.cpu().numpy()
                        x_list = x_list.cpu().numpy()
                        inter_P = inter_P.detach().cpu().numpy()
                        inter_u = inter_u.detach().cpu().numpy()
                        inter_v = inter_v.detach().cpu().numpy()
                        inter_T = inter_T.detach().cpu().numpy()
                        inter_q = inter_q.detach().cpu().numpy()
                        inter_rio = inter_rio.detach().cpu().numpy()

                        for id, (x, y) in enumerate(zip(x_list, y_list)):
                            y = int(y)
                            x = int(x)
                            result_u[y, x] = inter_u[id]
                            result_v[y, x] = inter_v[id]
                            result_P[y, x] = inter_P[id]
                            result_T[y, x] = inter_T[id]
                            result_q[y,x] = inter_q[id]
                            result_rio[y,x] = inter_rio[id]

                        result_file = os.path.join(vis_path,'train_results', '%d_result_u_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_u,result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_v_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_v, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_P_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_P, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_T_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_T, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_q_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_q, result_file)
                        result_file = os.path.join(vis_path,'train_results', '%d_result_rio_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_rio, result_file)
                    fps = time_metric.get_fps(log_step * batch_size)
                    fps=fps
                    time_metric.reset()
                    if local_rank==0:
                        print('=============================training==============================')
                        format_str = 'epoch:%d/%d,batch:%d/%d,iter:%d/%d,' % (
                            epoch, num_epoch, batch_id, len(train_dataloader), global_step,
                            len(train_dataloader) * num_epoch)
                        format_str = format_str + '%s:%f,' % ('train loss', train_loss)
                        for k, v in loss_dict.items():
                            format_str = format_str + '%s:%f,' % (k, v.item())
                        format_str = format_str + '%s:%03dh,%s:%03d,' % ('forecast', forecast_h_unnorm,
                                                                        'forecast_idx',forecast_idx)
                        format_str = format_str + '%s:%f,%s:%f' % ('grad',total_sum,'fps', fps)
                        print(format_str)
                        log_fp.writelines('%s\n' % format_str)
                        log_fp.flush()
                        if global_step>2000:
                            summary.add_scalar('training/total_loss', train_loss, global_step)
                            for k, v in loss_dict.items():
                                summary.add_scalar('training/%s' % k, v.detach(), global_step)
                            summary.add_scalar('training/margin_u_loss',margin_u_loss.detach(),global_step)
                            summary.add_scalar('training/margin_v_loss', margin_v_loss.detach(), global_step)
                            summary.add_scalar('training/margin_p_loss', margin_p_loss.detach(), global_step)
                            summary.add_scalar('training/margin_T_loss', margin_T_loss.detach(), global_step)
                            summary.add_scalar('training/margin_q_loss', margin_q_loss.detach(), global_step)
                            summary.add_scalar('training/margin_rio_loss', margin_rio_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_u_loss' % forecast_h_unnorm, margin_u_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_v_loss' % forecast_h_unnorm, margin_v_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_p_loss' % forecast_h_unnorm, margin_p_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_T_loss' % forecast_h_unnorm, margin_T_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_q_loss' % forecast_h_unnorm, margin_q_loss.detach(), global_step)
                            summary.add_scalar('training_f%03d/margin_rio_loss' % forecast_h_unnorm, margin_rio_loss.detach(), global_step)
                        print('=============================end==============================')
                    ###############validation
                    self.with_clip = True
                    try:
                        data=valid_iter.__next__()
                    except:
                        valid_iter=iter(valid_dataloader)
                        data = valid_iter.__next__()
                    field_data, margin_x, margin_y, margin_t, margin_data, margin_f,margin_input_data, \
                    inter_x, inter_y, inter_t, inter_data, inter_f, forecast_h_unnorm, input_file = data

                    field_data = field_data.to(device)
                    tmp_time_step = valid_dataset.input_time_step * valid_dataset.input_time_step_nums
                    forecast_h = forecast_h_unnorm // tmp_time_step * tmp_time_step / valid_dataset.forecast_time_period
                    forecast_h = forecast_h.float().to(device).unsqueeze(dim=0)
                    margin_x = margin_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_y = margin_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_t = margin_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    margin_f = margin_f.float().to(device).squeeze(dim=0)
                    margin_data = margin_data.float().to(device).squeeze(dim=0)
                    margin_input_data = margin_input_data.float().to(device).squeeze(dim=0)
                    #
                    inter_x = inter_x.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_y = inter_y.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_t = inter_t.float().to(device).squeeze(dim=0).unsqueeze(dim=1)
                    inter_f = inter_f[0].float().to(device).squeeze(dim=0)
                    inter_data = inter_data.float().to(device).squeeze(dim=0)
                    #
                    margin_input = self.encoding_coord(margin_x, margin_y, margin_t,
                                                       pred_t_span)
                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = self.physics_net.forward(field_data,
                                                                                                            margin_input,
                                                                                                            margin_input_data,
                                                                                                            forecast_h)
                    margin_loss = prediction_criterion(
                        torch.cat((margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio), dim=1),
                        margin_data).float()
                    margin_loss = margin_loss * loss_factor['margin_factor']
                    loss_dict = {'margin_loss': margin_loss}
                    if with_pde:
                        # x, y, t, f = self.place_grid(batch_size_inter, valid_dataset, pred_x_span, pred_y_span,
                        #                              pred_t_span,
                        #                              margin_t=False)
                        x=inter_x
                        y=inter_y
                        t=inter_t
                        f=inter_f
                        # x = torch.unsqueeze(x, dim=1)
                        # y = torch.unsqueeze(y, dim=1)
                        # t = torch.unsqueeze(t, dim=1)
                        x = x.requires_grad_(True)
                        y = y.requires_grad_(True)
                        t = t.requires_grad_(True)
                        inter_loss = self.place_one_batch(x, y, t, f, field_data,inter_data, forecast_h, pde_criterion,
                                                          loss_factor, global_step,local_rank, device,summary, 'inter',log_step=log_step).detach()

                        margin_x = margin_x.requires_grad_(True)
                        margin_y = margin_y.requires_grad_(True)
                        margin_t = margin_t.requires_grad_(True)
                        #
                        margin_pde_loss = self.place_one_batch(margin_x, margin_y, margin_t, margin_f, field_data,
                                                               margin_input_data,forecast_h,
                                                               pde_criterion, loss_factor
                                                               , global_step,local_rank,
                                                               device,summary, 'margin',log_step=log_step).detach()
                        loss_dict['inter_pde_loss'] = inter_loss.detach()
                        loss_dict['margin_pde_loss'] = margin_pde_loss.detach()

                    valid_loss = 0
                    for key, v in loss_dict.items():
                        valid_loss = valid_loss + v.detach()

                    margin_u, margin_v, margin_p, margin_T, margin_q, margin_rio = \
                        self.inverse_norm(margin_u.detach(), margin_v.detach(), margin_p.detach(), margin_T.detach(),
                                          margin_q.detach(), margin_rio.detach(),
                                          self.obs_norm_cfg)
                    label_u, label_v, label_p, label_T, label_q, label_rio = \
                        self.inverse_norm(margin_data[:, 0:1], margin_data[:, 1:2], margin_data[:, 2:3],
                                          margin_data[:, 3:4], margin_data[:, 4:5],
                                          margin_data[:, 5:6], self.obs_norm_cfg)
                    margin_u_loss = variable_criterion(margin_u, label_u).detach()
                    margin_v_loss = variable_criterion(margin_v, label_v).detach()
                    margin_p_loss = variable_criterion(margin_p, label_p).detach()
                    margin_T_loss = variable_criterion(margin_T, label_T).detach()
                    margin_q_loss = variable_criterion(margin_q, label_q).detach()
                    margin_rio_loss = variable_criterion(margin_rio, label_rio).detach()
                    if local_rank==0:
                        print('=============================validation==============================')
                        ref_h = forecast_h_unnorm.item()
                        # ref_h=np.round(ref_h,2)*train_dataset.forecast_time_period
                        forecast_idx = forecast_h.item() * valid_dataset.forecast_time_period
                        format_str = 'epoch:%d/%d,batch:%d/%d,iter:%d/%d,' % (
                            epoch, num_epoch, batch_id, len(train_dataloader), global_step,
                            len(train_dataloader) * num_epoch)
                        format_str = format_str + '%s:%f,' % ('valid loss', valid_loss)
                        for k, v in loss_dict.items():
                            format_str = format_str + '%s:%f,' % (k, v.item())
                        format_str = format_str + '%s:%03dh,%s:%03d,' % ('forecast', ref_h,
                                                                        'forecast_idx',forecast_idx)
                        format_str = format_str + '%s:%f' % ('fps', fps)
                        print(format_str)
                        log_fp.writelines('%s\n' % format_str)
                        log_fp.flush()
                        if global_step>2000:
                            summary.add_scalar('validation/total_loss', valid_loss, global_step)
                            for k, v in loss_dict.items():
                                summary.add_scalar('validation/%s' % k, v, global_step)
                            summary.add_scalar('validation/margin_u_loss', margin_u_loss.detach(), global_step)
                            summary.add_scalar('validation/margin_v_loss', margin_v_loss.detach(), global_step)
                            summary.add_scalar('validation/margin_p_loss', margin_p_loss.detach(), global_step)
                            summary.add_scalar('validation/margin_T_loss', margin_T_loss.detach(), global_step)
                            summary.add_scalar('validation/margin_q_loss', margin_q_loss.detach(), global_step)
                            summary.add_scalar('validation/margin_rio_loss', margin_rio_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_u_loss'%ref_h, margin_u_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_v_loss'%ref_h, margin_v_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_p_loss'%ref_h, margin_p_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_T_loss'%ref_h, margin_T_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_q_loss'%ref_h, margin_q_loss.detach(), global_step)
                            summary.add_scalar('validation_f%03d/margin_rio_loss'%ref_h, margin_rio_loss.detach(), global_step)
                        print('=============================end==============================')

                    if with_vis and local_rank==0:
                        self.with_clip=False
                        if with_pde:
                            time_id = \
                            np.random.randint(0, valid_dataset.input_time_step * valid_dataset.input_time_step_nums + 1,
                                              (1,))[0]
                        else:
                            time_id=0
                        x_list = []
                        y_list = []
                        t_list = []
                        for x in range(0, self.lon_size):
                            for y in range(0, self.lat_size):
                                x_list.append(x)
                                y_list.append(y)
                                t_list.append(time_id)
                        x_list, y_list, t_list, test_data, \
                        margin_f = valid_dataset.get_margin_grid(input_file[0],
                                                                                x_list, y_list, t_list) #* self.time_len
                        x_list=x_list.float().to(device)
                        y_list=y_list.float().to(device)
                        t_list=t_list.float().to(device)
                        test_data=test_data.float().to(device).squeeze(dim=0)
                        # t_list = torch.tensor(t_list) / 100
                        # input = torch.stack([x_list, y_list, t_list], dim=1).to(device)
                        input=self.encoding_coord(x_list,y_list,t_list,pred_t_span)
                        with torch.no_grad():
                            inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio = self.physics_net(field_data,input,
                                                                                                    test_data,forecast_h)
                        inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio= \
                            self.inverse_norm(inter_u, inter_v, inter_P,inter_T, inter_q,inter_rio,
                                              obs_norm_cfg=self.obs_norm_cfg)
                        # inter_p = inter_p * 255
                        result_v = np.zeros([self.lat_size, self.lon_size])
                        result_u = np.zeros([self.lat_size, self.lon_size])
                        result_P = np.zeros([self.lat_size, self.lon_size])
                        result_T = np.zeros([self.lat_size, self.lon_size])
                        result_q = np.zeros([self.lat_size, self.lon_size])
                        result_rio = np.zeros([self.lat_size, self.lon_size])

                        x_list=x_list/self.dx
                        y_list=y_list/self.dy
                        y_list = y_list.cpu().numpy()
                        x_list = x_list.cpu().numpy()
                        inter_P = inter_P.detach().cpu().numpy()
                        inter_u = inter_u.detach().cpu().numpy()
                        inter_v = inter_v.detach().cpu().numpy()
                        inter_T = inter_T.detach().cpu().numpy()
                        inter_q = inter_q.detach().cpu().numpy()
                        inter_rio = inter_rio.detach().cpu().numpy()

                        for id, (x, y) in enumerate(zip(x_list, y_list)):
                            y = int(y)
                            x = int(x)
                            result_u[y, x] = inter_u[id]
                            result_v[y, x] = inter_v[id]
                            result_P[y, x] = inter_P[id]
                            result_T[y, x] = inter_T[id]
                            result_q[y,x] = inter_q[id]
                            result_rio[y,x] = inter_rio[id]
                        forecast_h=forecast_h.item()
                        forecast_h=np.round(forecast_h,4)*train_dataset.forecast_time_period
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_u_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_u,result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_v_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_v, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_P_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_P, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_T_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_T, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_q_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_q, result_file)
                        result_file = os.path.join(vis_path,'valid_results', '%d_result_rio_f%03d.jpg' % (global_step,forecast_h))
                        vis_utils.forward(result_rio, result_file)

                    self.physics_net.train()
                # torch.distributed.barrier()
                # time.sleep(0.05)
                # if local_rank == 0: print('end')
            if epoch % save_step==0:
                if 'lr_schedule' in self.train_cfg.keys():
                    lr_schedule.step()
                    # summary.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'],
                    #                    global_step)
                if local_rank==0:
                    lr = optimizer.param_groups[0]['lr']
                    summary.add_scalar('learning_rate', lr, global_step)
                    self.save_model(checkpoint_path, epoch, global_step,  prefix='physics',
                                    dx=self.dx,dy=self.dy,dt=self.dt,pred_x_span=pred_x_span,
                                    pred_y_span=pred_y_span,pred_t_span=pred_t_span,
                                    label_time_step=time_step,input_variable_cfg=train_dataset.input_variable_cfg,
                                    input_time_step=train_dataset.input_time_step,
                                    input_time_step_nums=train_dataset.input_time_step_nums,
                                    obs_norm_cfg=self.obs_norm_cfg,
                                    start_time=train_dataset.start_time,end_time=train_dataset.end_time)


    def run_inference_interface(self,**kwargs):
        device = self.inference_cfg['device']
        img_size=self.inference_cfg['img_size']

        if isinstance(img_size, int) or isinstance(img_size, float):
            lat_size, lon_size = img_size, img_size
        elif (isinstance(img_size, list) or isinstance(img_size, tuple)) and len(img_size) == 2:
            lat_size, lon_size = img_size
        else:
            raise NotImplementedError

        dt = self.inference_cfg['dt']
        start_time=self.inference_cfg['start_time']
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d_%H_%M_%S')
        end_time = self.inference_cfg['end_time']
        end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d_%H_%M_%S')

        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path = kwargs['checkpoint_path']
        else:
            checkpoint_path = self.inference_cfg['checkpoints']['checkpoints_path']

        with_vis = self.inference_cfg['log']['with_vis']
        vis_path = self.inference_cfg['log']['vis_path']
        if (not os.path.exists(vis_path)) and with_vis:
            os.makedirs(vis_path)

        write_source=self.inference_cfg['log']['write_source']
        result_path=self.inference_cfg['log']['result_path']

        if (not os.path.exists(result_path)) and write_source:
            os.mkdir(result_path)
        vis_cfg = self.inference_cfg['log']['vis_downscale_cfg']
        if not (with_vis or write_source):
            print('warning: with_vis and write_source are both set False. No result will be saved.')
        export_variable=self.inference_cfg['log']['export_variable']
        self.physics_net.to(device)

        state_dict, current_epoch, global_step = self.load_model(checkpoint_path, prefix='physics')
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.physics_net.load_state_dict(state_dict['model'], strict=True)

            pred_t_span = self.gather_key_from_state('pred_t_span',state_dict,self.inference_cfg['pred_t_span'])
            self.obs_norm_cfg = self.gather_key_from_state('obs_norm_cfg',state_dict,self.obs_norm_cfg)
            time_step=self.gather_key_from_state('time_step',state_dict,3)
        else:
            raise NotImplementedError(checkpoint_path)


        pred_time=start_time+datetime.timedelta(seconds=pred_t_span-time_step*60*60)
        print('Predictable period:{0}->{1}'.format(start_time,pred_time))
        vis_utils=downscale_utils.VisUtils(**vis_cfg,img_size=img_size)
        ref_time=start_time
        time_id=0
        delta=(end_time-start_time).total_seconds()
        delta=list(range(int(delta/dt)))
        pbar=tqdm.tqdm(delta)

        while ref_time <= end_time:
            x_list = []
            y_list = []
            t_list = []
            for x in range(0, lon_size, 1):
                for y in range(0, lat_size, 1):
                    x_list.append(x)
                    y_list.append(y)
                    t_list.append(time_id * dt)
            x_list = torch.tensor(x_list)*self.dx
            y_list = torch.tensor(y_list)*self.dy
            t_list = torch.tensor(t_list)
            x_list=x_list.to(device)
            y_list = y_list.to(device)
            t_list = t_list.to(device)
            input = self.encoding_coord(x_list,y_list,t_list,self.pred_t_span)
            with torch.no_grad():
                inter_u, inter_v, inter_p, inter_T, inter_q,inter_rio = self.physics_net(input)
            inter_u, inter_v, inter_p, inter_T, inter_q, inter_rio = \
                self.inverse_norm(inter_u, inter_v, inter_p, inter_T, inter_q, inter_rio,
                                  obs_norm_cfg=self.obs_norm_cfg)
            pbar.update(1)
            time_id += 1
            ref_time = ref_time + datetime.timedelta(seconds=dt)
            result_p_img = np.zeros([lat_size, lon_size])
            result_u_img = np.zeros([lat_size, lon_size])
            result_v_img = np.zeros([lat_size, lon_size])
            result_T_img = np.zeros([lat_size, lon_size])
            result_rio_img = np.zeros([lat_size, lon_size])
            result_q_img = np.zeros([lat_size, lon_size])

            y_list = y_list.numpy() /self.dx
            x_list = x_list.numpy() /self.dy
            inter_u = inter_u.detach().cpu().numpy()
            inter_v = inter_v.detach().cpu().numpy()
            inter_p = inter_p.detach().cpu().numpy()
            inter_T = inter_T.detach().cpu().numpy()
            inter_rio = inter_rio.detach().cpu().numpy()
            inter_q = inter_q.detach().cpu().numpy()

            for id, (x, y) in enumerate(zip(x_list, y_list)):
                y = int(y)
                x = int(x)
                result_p_img[y, x] = inter_p[id]
                result_u_img[y, x] = inter_u[id]
                result_v_img[y, x] = inter_v[id]
                result_T_img[y, x] = inter_T[id]
                result_rio_img[y, x] = inter_rio[id]
                result_q_img[y, x] = inter_q[id]

            result_dict = {'U': result_u_img, 'V': result_v_img, 'P': result_p_img, 'T': result_T_img,
                           'RIO': result_rio_img, 'Q': result_q_img}

            for variable_name in export_variable:
                beijing_time=ref_time+datetime.timedelta(hours=6)
                result_img = result_dict[variable_name.upper()]
                if with_vis:
                    result_file = os.path.join(vis_path, '%s_%s_vis.jpg' % (beijing_time.strftime('%Y-%m-%d_%H_%M_%S'),variable_name))
                    vis_utils.forward(result_img, result_file)
                if write_source:
                    result_file = os.path.join(vis_path, '%s_%s.tiff' % (beijing_time.strftime('%Y-%m-%d_%H_%M_%S'),variable_name))
                    gdal_utils.save_full_image(result_file, result_img)

    def gather_key_from_state(self, k, state_dict: dict, default):
        if k in state_dict.keys():
            v = state_dict[k]
            print('find {0}, set {0} to {1}'.format(k, v))
            return v
        else:
            print('cannot find {0}, use it as default'.format(k))
            return default

    def print_key_args(self,**kwargs):
        for key,value in kwargs.items():
            print('{0}:{1}'.format(key,value))
