import torch
import argparse
import numpy as np
import mmcv
from DeepPhysiNet.interface.build import builder_models
import os
import shutil
import warnings
import zipfile
import glob
from DeepPhysiNet.utils import path_utils
# warnings.filterwarnings("ignore")
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
parse=argparse.ArgumentParser()

parse.add_argument('--config_file',default=r'configs/MetDNS_NCEP_wo_PDE_cfg.py',type=str)
parse.add_argument('--checkpoints_path',default=None,type=str)
parse.add_argument('--log_path',default=None,type=str)

def zip_codes(zip_file,zip_dir=('.','DeepDownScale','tool','exp')):
    zip_fp=zipfile.ZipFile(zip_file,'w',compression=zipfile.ZIP_DEFLATED)
    for dir in zip_dir:
        if os.getcwd()==os.path.abspath(dir):
            data_files=glob.glob(os.path.join(dir,'*.*'))
        else:
            data_files = glob.glob(os.path.join(dir, '**'),recursive=True)

        data_files=[x for x in data_files if os.path.isfile(x)]
        for data_file in data_files:
            zip_fp.write(data_file,compress_type=zipfile.ZIP_DEFLATED)
    zip_fp.close() 

if __name__=='__main__':
    args = parse.parse_args()
    print(args)
    cfg = mmcv.Config.fromfile(args.config_file)

    models=builder_models(**cfg['config'])
    run_args={}
    checkpoints_path= args.checkpoints_path if args.checkpoints_path is not None else cfg['config']['train_cfg']['checkpoints']['checkpoints_path']
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    shutil.copy(args.config_file,checkpoints_path)
    zip_file=os.path.join(checkpoints_path,'%s.zip'%path_utils.get_filename(checkpoints_path))
    zip_codes(zip_file)
    models.run_train_interface(checkpoint_path=args.checkpoints_path,
                                log_path=args.log_path)

