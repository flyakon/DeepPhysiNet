'''
@Project : deep-downscale 
@File    : downscale.py
@Author  : Wenyuan Li
@Date    : 2022/2/11 19:36 
@Desc    :  
'''

mode='train'
img_size=(145,257)
config=dict(
    name='InterfacePhysics',
    meta_cfg=dict(
        name='TransformerNet',
        enc_in=2405,
        c_out=256,
        d_model=256,
        n_heads=8,
        e_layers=4,
        d_ff=256,
        dropout=0.5,
        activation='gelu',
        output_attention=False
    ),
    net_cfg=dict(
        name='PhysicsNet',
        in_channels=192,
        hidden_channels=256,
        out_channels=1,
        token_num=155+4,
        learnable_token_num=256,
    ),

    variable_cfg=dict(
        #single_level
        pres=dict(name='PSFC',norm_factor=[89865.65002477072,13033.144877926803],norm_type='mean_norm',use_norm=True),
        t2=dict(name='t2',norm_factor=[284.6377185900894,15.672692198648798],norm_type='mean_norm',use_norm=True),
        u10=dict(name='u10',norm_factor=[0.3160574316187487,3.351126326454721],norm_type='mean_norm',use_norm=True),
        v10=dict(name='v10',norm_factor=[-0.014253187129747874,3.3603596038083645],norm_type='mean_norm',use_norm=True),
        q2=dict(name='q2',norm_factor=[0.007618763505692594,0.006144199452623363],norm_type='mean_norm',use_norm=True),
        rio=dict(name='rio',norm_factor=[1.0947008611668556,0.15032652292954654],norm_type='mean_norm',use_norm=True),
        #pressure_level
        GHT=dict(name='GHT',norm_factor=[
            [114.77246545150656, 771.4387290483181, 1477.9211924037004, 3066.5410091866233, 5699.21564876928],
            [78.84514334975856, 62.80230679864638, 53.68142954599702, 82.198609401067, 174.0655103756859]],
                 norm_type='mean_norm',use_norm=True),
        TT=dict(name='TT',norm_factor=[
            [291.8679412303275, 287.83199390277, 283.889157779038, 274.8786731408523, 259.0043477809711],
            [13.854616445675061, 13.399501237437411, 12.495739175598745, 10.891473152032383, 10.214928326417013]],
                norm_type='mean_norm',use_norm=True),
        UU=dict(name='UU',norm_factor=[
            [0.41332031537526026, 0.834938213915344, 1.865207683814799, 4.779482809007743, 10.125597561106549],
            [3.9687199045927044, 4.8036807628559774, 5.130202195055565, 6.140669757821302, 9.702883166443712]],
                norm_type='mean_norm',use_norm=True),
        VV=dict(name='VV',norm_factor=[
            [0.035755216965939404, 0.18368408301724948, 0.09977501517357934, -0.42035589580708466, -0.9547106399653137],
            [3.9183815477521424, 4.543953502929277, 4.730034382539942, 5.126140080824794, 6.746842619094734]],
                norm_type='mean_norm',use_norm=True),
        QQ=dict(name='QQ',norm_factor=[
            [0.00929879567731064, 0.007794286760000664, 0.00640619527691479, 0.004038364266386012, 0.0015411979441393073],
            [0.0067408698476321425, 0.005724667664620789, 0.004808302592428765, 0.003249943817624053, 0.0016106515214165957]],
               norm_type='mean_norm',use_norm=True),
    ),
    obs_norm_cfg=dict(
        pres=dict(name='PSFC',norm_factor=[89741.36105771353, 13296.749084125422],norm_type='mean_norm',
                  bound=[10000,500000],use_norm=True),
        t2=dict(name='t2',norm_factor=[283.58054561520305, 15.583177935722373],norm_type='mean_norm',
                bound=[50,500],use_norm=True),
        u10=dict(name='u10',norm_factor=[0.14507186950562942, 3.0050219075895894],norm_type='mean_norm',
                 bound=[-500,500],use_norm=True),
        v10=dict(name='v10',norm_factor=[-0.17325370241478535, 3.006602165591562],norm_type='mean_norm',
                 bound=[-500,500],use_norm=True),
        q2=dict(name='q2',norm_factor=[0.007909478276582905,0.006304067969976075],norm_type='mean_norm',
                bound=[1e-6, 10],use_norm=True),
        rio=dict(name='rio',norm_factor=[1.0966503643401704, 0.15166081218127583],norm_type='mean_norm',
                 bound=[1e-6,10],use_norm=True),
        total=dict(name='total',norm_factor=[
            [0.14507186950562942, -0.17325370241478535,  89741.36105771353,  283.58054561520305,
        0.007909478276582905,  1.0966503643401704],
            [3.0050219075895894, 3.006602165591562, 13296.749084125422, 15.583177935722373,
       0.006304067969976075, 0.15166081218127583]
            ],norm_type='mean_norm',use_norm=True)
    ),

    train_cfg=dict(
        batch_size=1,
        batch_size_inter=2048*2,
        device='cuda:0',
        num_epoch=201,
        num_workers=6,
        with_pde=True,
        lable_time_step=1,
        dx=27000,
        dy=27000,
        img_size=img_size,
        train_data=dict(
            input_path=r'/mnt/home/xx/training_data',
            label_path=r'/mnt/home/xx/training_labels',
            input_data_map_file=r'D/mnt/home/xx/train_data_map.pickle',
            constant_path=r'/mnt/home/xx/constant_variables',
            constant_variables=('landsea', 'elevation','lat','lon'),
            start_time=r'2008-01-01_00_00_00',
            end_time=r'2020-06-30_00_00_00',        #forecast start time
            in_coord_file=r'/mnt/home/xx/coord_1d.pickle',
            out_coord_file=r'/mnt/home/xx/coord_0p25d.pickle',
            input_time_step=6,     # 6 h
            input_time_step_nums=4, #6*4
            forecast_time_period=360,  #0~240 h
            label_time_step=1,
            label_img_size=img_size,
            label_batch_size=2048*10,
            batch_size_inter=2048*2,
            in_memory = False,
            auto_norm=False,
        ),
        valid_data=dict(
            input_path=r'/mnt/home/xx/training_data',
            label_path=r'/mnt/home/xx/training_labels',
            input_data_map_file=r'/mnt/home/xx/train_data_map.pickle',
            constant_path=r'/mnt/home/xx/constant_variables',
            constant_variables=('landsea', 'elevation','lat','lon'),
            start_time=r'2020-07-01_00_00_00',
            end_time=r'2020-12-31_00_00_00',        #forecast start time
            in_coord_file=r'/mnt/home/xx/coord_1d.pickle',
            out_coord_file=r'/mnt/home/xx/coord_0p25d.pickle',
            input_time_step=6,     # 6 h
            input_time_step_nums=4, #6*4
            forecast_time_period=360,  #0~240 h
            label_time_step=1,
            label_img_size=img_size,
            label_batch_size=2048*6,
            batch_size_inter=2048*3,
            in_memory = False,
            auto_norm=False
        ),
        losses=dict(
            pde_loss=dict(name='MSELoss'),
            prediction_loss=dict(name='WeightSmoothL1Loss',beta=0.1),
            # prediction_loss=dict(name='MSELoss'),
            loss_factor=dict(
                sample_factor=1.e6,
                margin_factor=1.e6,
                motion_u_factor=1.e3,
                motion_v_factor=1.e3,
                continuous_factor=1.e10,
                energy_factor=1e1,
                vapor_factor=1.e14,
                gas_factor=1.e-7
                             )
        ),

        optimizer=dict(
            name='Adam',
            lr=1e-4,
            weight_decay=1e-4
        ),
        checkpoints=dict(
            checkpoints_path=r'checkpoints/DeepPhysiNet',
            save_step=1,
        ),
        lr_schedule=dict(
            name='CosineAnnealingLR',
            T_max=5,
            eta_min=5e-6,
            verbose=True
        ),
        log=dict(
            log_path=r'log/DeepPhysiNet',
            log_step=100,
            with_vis=True,
            vis_path=r'../results/DeepPhysiNet',
            vis_downscale_cfg=dict(
                coord_file=r'/mnt/home/xx/coord_0p25d.pickle',
                project_dict=dict(name='LatLon')
            )

        ),
    ),

    test_cfg=dict(
        batch_size=1,
        device='cuda:0',
        num_epoch=105,
        num_workers=0,

        test_data=dict(
            input_path=r'',
            label_path=r'',
            input_format='*.tiff',
            label_format='*.tiff',
            in_memory=False,
            time_span=32,
            time_step=3,
            label_factor=(800, 1100)
        ),
        checkpoints=dict(
            checkpoints_path=r'checkpoints',

        ),
        log=dict(
            result_file=r'',
            with_vis=False,
            vis_path=r''
        ),
    ),
    inference_cfg=dict(
        batch_size=1,
        device='cuda:0',
        num_epoch=105,
        num_workers=0,
        dt=60*60,
        img_size=128,
        pred_t_span=-1,
        start_time=r'2022-03-25_00_00_00',
        end_time=r'2022-03-31_00_00_00',
        checkpoints=dict(
            checkpoints_path=r'checkpoints\checkpoints/DeepPhysiNet',
        ),
        log=dict(
            with_vis=True,
            vis_path=r'',
            result_path=r'',
            write_source=False,
            export_variable=['T'],

            vis_downscale_cfg=dict(
                coord_file=r'',
                project_dict=dict(name='Mercator', stand_lon=110.0, moad_cen_lat=30.0, truelat1=30,
                                  truelat2=60.0, pole_lat=90.0, pole_lon=0.0)
            )

        )
        )
)

