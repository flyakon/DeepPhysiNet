'''李汶原 20180804
gdal读取图像的辅助工具库
'''

from pyproj import Proj
import numpy as np
import os
try:
    from osgeo import osr
    from osgeo import gdal
except:
    import gdal
    import osr

from bs4 import BeautifulSoup

def convt_geo(x,y,proj='utm', zone_id=42, ellps='WGS84', south=False,inverse=True,**kwargs):

    p = Proj(proj=proj, zone=zone_id, ellps=ellps, south=south, preserve_units=False,**kwargs)
    lon,lat= p(x, y, inverse=inverse)
    return lon,lat

def get_image_shape(img_path):
    '''
    获取图像的尺寸，格式为(height，width，bands)
    :param img_path: 
    :return: 
    '''

    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    del dataset
    return im_height,im_width,im_bands

#待补充
# def save_image(img_path,img,width_offset,height_offset,width,height,
#                geoTranfsorm=None,proj=None,data_format='GDAL_FORMAT'):
#     '''
#     保存图像
#     :param img_path: 保存的路径
#     :param img:
#     :param geoTranfsorm:
#     :param proj:
#     :return:
#     '''
#     if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
#         raise Exception('data_format参数错误')
#     if 'uint8' in img.dtype.name:
#         datatype=gdal.GDT_Byte
#     elif 'int16' in img.dtype.name:
#         datatype=gdal.GDT_CInt16
#     else:
#         datatype=gdal.GDT_Float32
#     if len(img.shape)==3:
#         if data_format=='NUMPY_FORMAT':
#             img = np.swapaxes(img, 1, 2)
#             img = np.swapaxes(img, 0, 1)
#         im_bands,im_height,im_width=img.shape
#     elif len(img.shape)==2:
#         img=np.array([img])
#         im_bands,im_height, im_width = img.shape
#     else:
#         im_bands,(im_height,im_width)=1,img.shape
#
#     driver=gdal.GetDriverByName("GTIFF")
#     # if os.path.exists(img_path):
#     #     dataset=driver.Create(img_path,im_width,im_height,im_bands,datatype)
#     dataset = gdal.Open(img_path)
#     if dataset is None:
#         print("文件%s无法打开" % img_path)
#         exit(-1)
#     full_height,full_width,_=get_image_shape(img_path)
#
#     if width_offset+width>full_width:
#         block_width=full_width-width_offset
#     if height_offset+height>full_height:
#         block_height=full_height-height_offset
#     if geoTranfsorm:
#         dataset.SetGeoTransform(geoTranfsorm)
#     if proj:
#         dataset.SetProjection(proj)
#     for i in range(im_bands):
#         dataset.GetRasterBand(i+1).WriteArray(img[i],width_offset,height_offset)

def save_full_image(img_path,img,geoTranfsorm=None,proj=None,data_format='GDAL_FORMAT',**kwargs):
    '''
    保存图像
    :param img_path: 保存的路径
    :param img: 
    :param geoTranfsorm: 
    :param proj: 
    :return: 
    '''
    if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    if 'uint8' in img.dtype.name:
        datatype=gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype=gdal.GDT_CInt16
    # elif 'float64' in img.dtype.name:
    #     datatype = gdal.GDT_Float64
    else:
        datatype=gdal.GDT_Float64
    if len(img.shape)==3:
        if data_format=='NUMPY_FORMAT':
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 1)
        im_bands,im_height,im_width=img.shape
    elif len(img.shape)==2:
        img=np.array([img])
        im_bands,im_height, im_width = img.shape
    else:
        im_bands,(im_height,im_width)=1,img.shape

    driver=gdal.GetDriverByName("GTIFF")
    dataset=driver.Create(img_path,im_width,im_height,im_bands,datatype,**kwargs)
    if geoTranfsorm:
        dataset.SetGeoTransform(geoTranfsorm)
    if proj:
        dataset.SetProjection(proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(img[i])

def gdal_copy(src_file,dst_file,options=None):
    driver = gdal.GetDriverByName("GTIFF")
    dataset=gdal.Open(src_file)
    driver.CreateCopy(dst_file,dataset,strict=1,options=options)


def read_full_image(img_path,scale_factor=1,as_rgb=False,
               data_format='GDAL_FORMAT',normalize=False,normalize_factor=16,band_idx=None):
    '''
    一次读取整张图片
    :param img_path: 
    :param scale_factor: 
    :param as_rgb: 
    :param data_format: 
    :return: 
    '''
    im_height,im_width,_=get_image_shape(img_path)
    img=read_image(img_path,0,0,im_width,im_height,scale_factor,as_rgb,data_format,normalize=normalize,
                   normalize_factor=normalize_factor,band_idx=band_idx)
    return img

def read_image(img_path,width_offset,height_offset,width,height,scale_factor=1,as_rgb=True,
               data_format='GDAL_FORMAT',normalize=True,normalize_factor=16,band_idx=None):
    '''
    读取图片,支持分块读取,若读取的尺寸超过图像的实际尺寸，则在边界补0
    :param img_path: 要读取的图片的路径
    :param width_offset: x方向的偏移量
    :param height_offset: y方向上的偏移量
    :param width: 要读取的图像快的宽度
    :param height: 要读取的图像快的高度
    :param scale_factor:缩放比例
    :param as_rgb:是否将灰度图转化为rgb图
    :param data_format:返回结果的格式,有两种取值：'GDAL_FORMAT','NUMPY_FORMAT'
                    'GDAL_FORMAT':返回图像的shape为'(bands,height,width)'
                    'NUMPY_FORMAT':返回图像的尺寸为(height,width,bands)
                    每种格式下返回的图像的shape长度都为3
    :return: 
    '''
    if data_format not in ['GDAL_FORMAT','NUMPY_FORMAT']:
        raise Exception('data_format参数错误')
    dataset=gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开"%img_path)
        exit(-1)

    im_width = dataset.RasterXSize  # 图像的列数
    im_height = dataset.RasterYSize
    if band_idx is None:
        im_bands = dataset.RasterCount
        band_idx=list(range(1,im_bands+1))
    else:
        im_bands=len(band_idx)
    scale_width = int(width / scale_factor)
    scale_height = int(height / scale_factor)
    #判断索引是否越界，只读取不越界部分的图像，其余部分补0
    block_width=width
    block_height=height
    if width_offset+width>im_width:
        block_width=im_width-width_offset
    if height_offset+height>im_height:
        block_height=im_height-height_offset
    scale_block_width = int(block_width / scale_factor)
    scale_block_height = int(block_height / scale_factor)
    im_data=np.zeros((im_bands,scale_block_height,scale_block_width),dtype=np.float64)
    for i,idx in enumerate(band_idx):
        band=dataset.GetRasterBand(idx)
        im_data[i]=band.ReadAsArray(width_offset,height_offset,block_width,block_height,
                                scale_block_width,scale_block_height)
    if im_bands==1 and as_rgb:
        im_data = np.tile(im_data,(3,1,1))
    elif im_bands>=4 and as_rgb:
        im_data = im_data[0:3,:,:]

    if normalize:
        if isinstance(normalize_factor,int):
            for i in range(len(im_data)):
                im_data[i]= (im_data[i].astype(np.float32)/ normalize_factor).astype(np.uint16)

        elif len(normalize_factor)==2:
            for i in range(len(im_data)):
                tmp=im_data[i].astype(np.float32)
                tmp=(tmp-normalize_factor[0])/(normalize_factor[1]-normalize_factor[0])
                tmp=np.clip(tmp,0.,1.)
                tmp=tmp*255
                im_data[i]=tmp.astype(np.uint16)
        else:
            raise NotImplementedError


        im_data=im_data.astype(np.uint8)  #此时的shape为(bands,height,width)?待验证height和width的顺序

    if width!=block_width or height!=block_height:
        im_data=np.pad(im_data,((0,0),(0,scale_height-scale_block_height),(0,scale_width-scale_block_width)),mode='constant')

    if data_format=='NUMPY_FORMAT':
        im_data=np.swapaxes(im_data,0,1)
        im_data=np.swapaxes(im_data,1,2)
    del dataset
    return im_data

def get_geoTransform(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    geotransform=dataset.GetGeoTransform()
    return geotransform

def get_transforms_xml(img_file,xml_file):
    im_height,im_width,_=get_image_shape(img_file)
    file = open(xml_file, encoding='utf-8').read()
    soup = BeautifulSoup(file, 'xml')
    get_transforms=[-1.]*6
    get_transforms[0]=float(soup.find('TopLeftLongitude').text)
    get_transforms[3] = float(soup.find('TopLeftLatitude').text)
    xmin = float(soup.find('TopLeftLongitude').text)
    ymin = float(soup.find('TopLeftLatitude').text)

    x = float(soup.find('TopRightLongitude').text)
    y = float(soup.find('TopRightLatitude').text)
    get_transforms[1]=(x - xmin)/im_width
    get_transforms[4] = (y - ymin) / im_width

    x = float(soup.find('BottomLeftLongitude').text)
    y = float(soup.find('BottomLeftLatitude').text)

    get_transforms[2] = (x - xmin) / im_height
    get_transforms[5] = (y - ymin) / im_height

    return tuple(get_transforms)


def get_transforms_rpb(img_file,result_file,exe_path):
    im_height,im_width,_=get_image_shape(img_file)
    img_file=os.path.abspath(img_file)
    exe_path=os.path.abspath(exe_path)
    result_file=os.path.abspath(result_file)
    get_transforms = [-1.] * 6

    ##读取左上坐标
    offset_height,offset_width=0,0
    cmd_line='%s %s %s %d %d'%(exe_path,img_file,result_file,offset_width,offset_height)
    os.system(cmd_line)
    # print(cmd_line)
    lonlat_list=np.loadtxt(result_file)
    if lonlat_list[-1]==0:
        raise Exception("左上坐标解析错误")
    get_transforms[0]=lonlat_list[0]
    get_transforms[3] = lonlat_list[1]
    xmin = lonlat_list[0]
    ymin = lonlat_list[1]

    ##读取右上的坐标
    for i in np.arange(0,1,0.1,dtype=np.float):
        offset_width, offset_height = int(im_width*(1-i)), 0
        cmd_line = '%s %s %s %d %d' % (exe_path, img_file, result_file, offset_width, offset_height)
        os.system(cmd_line)
        # print(cmd_line)
        lonlat_list = np.loadtxt(result_file)
        if lonlat_list[-1]>0:
            x = lonlat_list[0]
            y = lonlat_list[1]
            get_transforms[1]=(x - xmin)/offset_width
            get_transforms[4] = (y - ymin) / offset_width
            break
    else:
        raise Exception("无法转换")

    ##解析左下
    for i in np.arange(0,1,0.1,dtype=np.float):
        offset_width, offset_height = 0, int(im_height*(1-i))
        cmd_line = '%s %s %s %d %d' % (exe_path, img_file, result_file, offset_width, offset_height)
        os.system(cmd_line)
        # print(cmd_line)
        lonlat_list = np.loadtxt(result_file)
        if lonlat_list[-1]>0:
            x = lonlat_list[0]
            y = lonlat_list[1]
            get_transforms[2] = (x - xmin) / offset_height
            get_transforms[5] = (y - ymin) / offset_height
            break
    else:
        raise Exception("无法转换")


    return tuple(get_transforms)

def get_projection(img_path):
    dataset = gdal.Open(img_path)
    if dataset is None:
        print("文件%s无法打开" % img_path)
        exit(-1)
    projection=dataset.GetProjection()
    return projection

def get_dataset(img_file):
    dataset = gdal.Open(img_file)
    if dataset is None:
        print("文件%s无法打开" % img_file)
        exit(-1)
    return dataset

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs



def geo2lonlat( x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(4326)
    geosrs = osr.SpatialReference()
    geosrs.ImportFromEPSG(3857)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def conver_transforms(dataset,geotransform):
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    projection=dataset.GetProjection()
    id_s = projection.find('UTM zone ') + len('UTM zone ')
    id_e = projection.find('GEOGCS') - 2
    zone = projection[id_s:id_e]
    zone_id = int(zone[-3:-1])
    south = False if zone[-1] == 'N' else True
    # p = Proj(proj='utm', zone=zone_id, ellps='WGS84', south=south, preserve_units=False)
    p = Proj(proj='utm', ellps='WGS84')
    # print('lon=%8.3f lat=%5.3f' % p(x, y, inverse=True))
    min_lon,min_lat=p(geotransform[0],geotransform[3],inverse=False)
    xmax=geotransform[0]+geotransform[1]*(im_width-1)
    ymax = geotransform[3] + geotransform[5] * (im_height-1)
    max_lon,max_lat=p(xmax,ymax,inverse=False)
    # print(max_lon,max_lat)
    lon_delta=(max_lon-min_lon)/(im_width-1)
    lat_delta=(max_lat-min_lat)/(im_height-1)
    return [min_lon,lon_delta,0,min_lat,0,lat_delta]

def lonlat2geo( lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''


    prosrs = osr.SpatialReference()
    prosrs.ImportFromEPSG(3857)
    geosrs = osr.SpatialReference()
    geosrs.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def swap_band(img):

    result_img = np.zeros_like(img)
    result_img[ :, :,0] = img[:, :,2]
    result_img[:, :,2] = img[:, :,0]
    result_img[ :, :,1] = img[ :, :,1]
    return result_img


