'''
@Project : deep-downscale 
@File    : builder.py
@Author  : Wenyuan Li
@Date    : 2022/2/11 20:06 
@Desc    :  
'''


from .physics_net import PhysicsNet

model_dict={'PhysicsNet':PhysicsNet,
            }

def build(name='PhysicsNet',**kwargs):
    if name in model_dict.keys():
        return model_dict[name](**kwargs)
    else:
        raise NotImplementedError(r'''{0} is not an available values. \
                                          Please choose one of the available values in
                                           [vgg11,vgg16]'''.format(name))
