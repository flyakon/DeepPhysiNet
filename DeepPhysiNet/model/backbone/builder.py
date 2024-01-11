# from .resnet import get_resnet
from .resnet import get_resnet
import torch

def build_backbone(name='resnet50',**kwargs):
    if name.startswith('resnet'):
        # model=get_resnet(name,**kwargs)
        return get_resnet(name,**kwargs)
    else:
        raise NotImplementedError(r'''{0} is not an available values. \
                                          Please choose one of the available values in
                                           [vgg11,vgg16]'''.format(name))
