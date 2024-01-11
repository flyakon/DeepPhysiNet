'''
@Project : crop-monitoring 
@File    : build.py
@Author  : Wenyuan Li
@Date    : 2021/12/30 21:43 
@Desc    :  
'''


from .interface_physics import InterfacePhysics
interface_dict={
             'InterfacePhysics':InterfacePhysics,
            }


def builder_models(name='InterfaceDownScale',**kwargs):
    if name in interface_dict.keys():
        return interface_dict[name](**kwargs)
    else:
        raise NotImplementedError('{0} not in availables values.'.format(name))