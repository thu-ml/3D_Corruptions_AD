import mmcv
import warnings
from copy import deepcopy

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


import warnings
from copy import deepcopy

import mmcv
import numpy as np


from .Camera_corruptions import ImagePointAddSun,ImageAddSnow,ImageAddFog,ImageAddRain,ImageAddGaussianNoise,ImageAddImpulseNoise,ImageAddUniformNoise
from .Camera_corruptions import ImageBBoxOperation
from .Camera_corruptions import ImageMotionBlurFrontBack, ImageMotionBlurLeftRight
from .Camera_corruptions import ImageBBoxMotionBlurFrontBack, ImageBBoxMotionBlurLeftRight

import torch
from .LiDAR_corruptions import gaussian_noise,lidar_crosstalk_noise,density_dec_global,density_dec_local,cutout_local,uniform_noise,background_noise,upsampling,impulse_noise,layer_del,scene_glare_noise,cutout_bbox,density_dec_bbox,gaussian_noise_bbox,scale_bbox,shear_bbox,FFD_bbox,moving_noise_bbox,uniform_noise_bbox,upsampling_bbox,impulse_noise_bbox,rotation_bbox,rain_sim_lidar,fog_sim_lidar,snow_sim_lidar,spatial_alignment_noise,fulltrajectory_noise, temporal_alignment_noise,filter_point_by_angle


def format_list_float_06(l) -> None:
    for index, value in enumerate(l):
        l[index] = float('%.6f' % value)
    return l

def load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    points = np.fromfile(pts_filename, dtype=np.float32)
    return points

def remove_close(points, radius=1.0):
    """Removes point too close within a certain radius from origin.

    Args:
        points (np.ndarray): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray: Points after removing.
    """
    if isinstance(points, np.ndarray):
        points_numpy = points
    elif isinstance(points, BasePoints):
        points_numpy = points.tensor.numpy()
    else:
        raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]



@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict


# 朱子健 add

from .img_corrupt import ImagePointAddSun,ImageAddSunMono

from .img_corrupt import ImageAddSnow,ImageAddFog,ImageAddRain,ImageAddGaussianNoise,ImageAddImpulseNoise,ImageAddUniformNoise
from .img_corrupt import ImageBBoxOperation, ImageBBoxOperationMono
from .img_corrupt import ImageMotionBlurFrontBack, ImageMotionBlurLeftRight

from .img_corrupt import ImageBBoxMotionBlurFrontBack, ImageBBoxMotionBlurLeftRight
from .img_corrupt import ImageBBoxMotionBlurFrontBackMono, ImageBBoxMotionBlurLeftRightMono



# 朱子健 add
@PIPELINES.register_module()
class CorruptionMethods(object):
    """Test-time augmentation with corruptions.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(self,
                 corruption_severity_dict=
                    {
                        # just for demo
                        'sun_sim':2,
                    },
                 ):


        # 能作为全局设定存在的，应是指定：
        # 1.用什么corruption. 2.扰动的程度

        self.corruption_severity_dict = corruption_severity_dict

        if 'sun_sim' in self.corruption_severity_dict:
            # 注意这个是加点云和图像双重加噪，或mono的纯图像加噪
            np.random.seed(2022)
            severity = self.corruption_severity_dict['sun_sim']
            self.sun_sim = ImagePointAddSun(severity)
            self.sun_sim_mono = ImageAddSunMono(severity)

        if 'snow_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['snow_sim']
            self.snow_sim = ImageAddSnow(severity, seed=2022)

        if 'fog_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['fog_sim']
            self.fog_sim = ImageAddFog(severity, seed=2022)
            
        if 'rain_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['rain_sim']
            self.rain_sim = ImageAddRain(severity, seed=2022)

        # if 'motion_sim' in self.corruption_severity_dict:
        #     # 注意这个只是加图像噪声，没有点云干扰
        #     severity = self.corruption_severity_dict['motion_sim']
        #     self.motion_sim = ImageMotionBlurFrontBack(severity)

        if 'motion_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['motion_sim']
            self.motion_sim_leftright = ImageMotionBlurFrontBack(severity)
            self.motion_sim_frontback = ImageMotionBlurLeftRight(severity)
        
        if 'gauss_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['gauss_sim']
            self.gauss_sim = ImageAddGaussianNoise(severity, seed=2022)
        
        if 'impulse_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['impulse_sim']
            self.impulse_sim = ImageAddImpulseNoise(severity, seed=2022)

        if 'uniform_sim' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['uniform_sim']
            self.uniform_sim = ImageAddUniformNoise(severity)

        # if 'object_motion_sim' in self.corruption_severity_dict:
        #     # 注意这个只是加图像噪声，没有点云干扰
        #     severity = self.corruption_severity_dict['object_motion_sim']
        #     # for kitti and nus
        #     self.object_motion_sim_frontback = ImageBBoxMotionBlurFrontBack(severity)
        #     self.object_motion_sim_leftright = ImageBBoxMotionBlurLeftRight(severity)

        ######################################################
        # object motion blur
        ######################################################
        if 'object_motion_sim' in self.corruption_severity_dict:
            # for kitti and nus
            severity = self.corruption_severity_dict['object_motion_sim']
            self.object_motion_sim_frontback = ImageBBoxMotionBlurFrontBack(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.object_motion_sim_leftright = ImageBBoxMotionBlurLeftRight(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.object_motion_sim_frontback_mono = ImageBBoxMotionBlurFrontBackMono(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )
            self.object_motion_sim_leftright_mono = ImageBBoxMotionBlurLeftRightMono(
                severity=severity,
                corrput_list=[0.02 * i for i in range(1, 6)],
            )

        #######################################################
        # bbox operation
        #######################################################
        if 'bbox_shear' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['bbox_shear']
            self.bbox_shear = ImageBBoxOperation(severity)
            self.bbox_shear_mono = ImageBBoxOperationMono(severity)

        if 'bbox_rotate' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['bbox_rotate']
            self.bbox_rotate = ImageBBoxOperation(severity)
            self.bbox_rotate_mono = ImageBBoxOperationMono(severity)

        if 'bbox_scale' in self.corruption_severity_dict:
            # 注意这个只是加图像噪声，没有点云干扰
            severity = self.corruption_severity_dict['bbox_scale']
            self.bbox_scale = ImageBBoxOperation(severity)
            self.bbox_scale_mono = ImageBBoxOperationMono(severity)

        # 朱子健 tips: 点云扰动也可以放在这里。参考太阳干扰
        # 注意输入是
        # points_tensor = results['points'].tensor
        # 输出是
        # results['points'].tensor = points_aug




    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """

        if 'sun_sim' in self.corruption_severity_dict:

            # 双数据集  公共部分
            img_bgr_255_np_uint8 = results['img']  # nus:各论各的list / kitti: nparray

            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus:各论各的list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono 数据集
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('zzj: no lidar2img or cam_intrinsic found!')

            if not use_mono_dataset:
                points_tensor = results['points'].tensor
                # nus 和 kitti 不同部分
                if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                    # 判断是否是 nus 数据集
                    # 太阳只需要加一个
                    '''
                    nuscenes:
                    0    CAM_FRONT,
                    1    CAM_FRONT_RIGHT,
                    2    CAM_FRONT_LEFT,
                    3    CAM_BACK,
                    4    CAM_BACK_LEFT,
                    5    CAM_BACK_RIGHT
                    '''
                    img_rgb_255_np_uint8_0 = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
                    lidar2img_0 = lidar2img[0]
                    image_aug_rgb_0, points_aug = self.sun_sim(
                        image=img_rgb_255_np_uint8_0,
                        points=points_tensor,
                        lidar2img=lidar2img_0,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
                    img_bgr_255_np_uint8[0] = image_aug_bgr_0
                    results['img'] = img_bgr_255_np_uint8
                    results['points'].tensor = points_aug

                elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                    # 判断是否是 waymo 数据集
                    # 太阳只需要加一个
                    '''
                    nuscenes:
                    0    CAM_FRONT,
                    1    CAM_FRONT_RIGHT,
                    2    CAM_FRONT_LEFT,
                    3    CAM_BACK,
                    4    CAM_BACK_LEFT,
                    5    CAM_BACK_RIGHT
                    '''
                    img_rgb_255_np_uint8_0 = img_bgr_255_np_uint8[0][:, :, [2, 1, 0]]
                    lidar2img_0 = lidar2img[0]
                    image_aug_rgb_0, points_aug = self.sun_sim(
                        image=img_rgb_255_np_uint8_0,
                        points=points_tensor,
                        lidar2img=lidar2img_0,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
                    img_bgr_255_np_uint8[0] = image_aug_bgr_0
                    results['img'] = img_bgr_255_np_uint8
                    results['points'].tensor = points_aug

                else:
                    # 判断是 kitti 数据集
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb, points_aug = self.sun_sim(
                        image=img_rgb_255_np_uint8,
                        points=points_tensor,
                        lidar2img=lidar2img,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr
                    results['points'].tensor = points_aug
            else:
                # mono 数据集 nus kitti都只有一张图
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                image_aug_rgb = self.sun_sim_mono(
                    image=img_rgb_255_np_uint8,
                    # watch_img=True,
                    # file_path='2.jpg'
                )
                image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                results['img'] = image_aug_bgr

        # if 'snow_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     # points_tensor = results['points'].tensor
        #     # lidar2img = results['lidar2img']
        #     image_aug_rgb = self.snow_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr

        # if 'fog_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     # points_tensor = results['points'].tensor
        #     # lidar2img = results['lidar2img']
        #     image_aug_rgb = self.fog_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr

        # if 'rain_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     # points_tensor = results['points'].tensor
        #     # lidar2img = results['lidar2img']
        #     image_aug_rgb = self.rain_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr

        #kcx 适配6图

        if 'snow_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.snow_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.snow_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.snow_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr

        if 'fog_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.fog_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.fog_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.fog_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr


        if 'rain_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.rain_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.rain_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.rain_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr


        if 'motion_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                # 判断是否是 nus 数据集
                '''
                nuscenes:
                0    CAM_FRONT,
                1    CAM_FRONT_RIGHT,
                2    CAM_FRONT_LEFT,
                3    CAM_BACK,
                4    CAM_BACK_LEFT,
                5    CAM_BACK_RIGHT
                '''
                image_aug_bgr = []
                for i in range(6):
                    img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                    if i % 3 == 0:
                        image_aug_rgb_i = self.motion_sim_frontback(
                            image=img_rgb_255_np_uint8_i,
                            # watch_img=True,
                            # file_path='2.png'
                        )
                    else:
                        image_aug_rgb_i = self.motion_sim_leftright(
                            image=img_rgb_255_np_uint8_i,
                            # watch_img=True,
                            # file_path='3.png'
                        )
                    image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                    image_aug_bgr.append(image_aug_bgr_i)
                results['img'] = image_aug_bgr
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                # 判断是否是 nus 数据集
                '''
                nuscenes:
                0    CAM_FRONT,
                1    CAM_FRONT_RIGHT,
                2    CAM_FRONT_LEFT,
                3    CAM_BACK,
                4    CAM_BACK_LEFT,
                5    CAM_BACK_RIGHT
                '''
                image_aug_bgr = []
                for i in range(5):
                    img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                    if i == 0:
                        image_aug_rgb_i = self.motion_sim_frontback(
                            image=img_rgb_255_np_uint8_i,
                            # watch_img=True,
                            # file_path='2.png'
                        )
                    else:
                        image_aug_rgb_i = self.motion_sim_leftright(
                            image=img_rgb_255_np_uint8_i,
                            # watch_img=True,
                            # file_path='3.png'
                        )
                    image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                    image_aug_bgr.append(image_aug_bgr_i)
                results['img'] = image_aug_bgr

            else:
                # 判断是 kitti 数据集
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                # print('!!!!!!!!-----------------------------------!!!!!!!!!!')
                # print('attention the front-back image or the leftright image')
                # print(' different in kitti and nus  ')
                # print('!!!!!!!!-----------------------------------!!!!!!!!!!')

                image_aug_rgb = self.motion_sim_frontback(
                    image=img_rgb_255_np_uint8,
                    # watch_img=True,
                    # file_path='2.png'
                )
                image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                results['img'] = image_aug_bgr
        
        ###kcx noise kitti单图版

        # if 'gauss_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     image_aug_rgb = self.gauss_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr


        # if 'impulse_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     image_aug_rgb = self.impulse_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr
        
        # if 'uniform_sim' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
        #     image_aug_rgb = self.uniform_sim(
        #         image=img_rgb_255_np_uint8
        #         )
        #     image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
        #     results['img'] = image_aug_bgr



        ###kcx noise适配kitti/nus

        if 'gauss_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.gauss_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.gauss_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.gauss_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr

        if 'impulse_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.impulse_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.impulse_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.impulse_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr

        if 'uniform_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                img = []
                for i in range(6):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.uniform_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                img = []
                for i in range(5):
                    new_np = img_bgr_255_np_uint8[i][:,:,[2,1,0]]
                    image_aug_rgb = self.uniform_sim(
                    image = new_np
                    )
                    image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                    img.append(image_aug_bgr)
                results['img'] = img
            else:
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
                image_aug_rgb = self.uniform_sim(
                    image=img_rgb_255_np_uint8
                    )
                image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
                results['img'] = image_aug_bgr
        
        

        if 'object_motion_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            # points_tensor = results['points'].tensor
            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus:各论各的list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono 数据集
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('zzj: no lidar2img or cam_intrinsic found!')
            #kcx 避免为空：
            
            bboxes_corners = results['gt_bboxes_3d'].corners
            
            bboxes_centers = results['gt_bboxes_3d'].center


            if type(bboxes_corners) == int:
                print(0)

            if type(bboxes_corners) != int:

                if not use_mono_dataset:
                    if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(6):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]

                            if i % 3 == 0:
                                image_aug_rgb_i = self.object_motion_sim_frontback(
                                    image=img_rgb_255_np_uint8_i,
                                    bboxes_centers=bboxes_centers,
                                    bboxes_corners=bboxes_corners,
                                    lidar2img=lidar2img_i,
                                    # watch_img=True,
                                    # file_path='2.jpg'
                                )
                            else:
                                image_aug_rgb_i = self.object_motion_sim_leftright(
                                    image=img_rgb_255_np_uint8_i,
                                    bboxes_centers=bboxes_centers,
                                    bboxes_corners=bboxes_corners,
                                    lidar2img=lidar2img_i,
                                    # watch_img=True,
                                    # file_path='2.jpg'
                                )
                                # print('object_motion_sim_leftright:', time_inter)
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                        results['img'] = image_aug_bgr

                    elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(5):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]

                            # if i % 3 == 0:
                            if i == 0:
                                image_aug_rgb_i = self.object_motion_sim_frontback(
                                    image=img_rgb_255_np_uint8_i,
                                    bboxes_centers=bboxes_centers,
                                    bboxes_corners=bboxes_corners,
                                    lidar2img=lidar2img_i,
                                    # watch_img=True,
                                    # file_path='2.jpg'
                                )
                            else:
                                image_aug_rgb_i = self.object_motion_sim_leftright(
                                    image=img_rgb_255_np_uint8_i,
                                    bboxes_centers=bboxes_centers,
                                    bboxes_corners=bboxes_corners,
                                    lidar2img=lidar2img_i,
                                    # watch_img=True,
                                    # file_path='2.jpg'
                                )
                                # print('object_motion_sim_leftright:', time_inter)
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                        results['img'] = image_aug_bgr

                    else:
                        # 判断是 kitti 数据集
                        img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                        # points_tensor = results['points'].tensor
                        lidar2img = results['lidar2img']
                        bboxes_corners = results['gt_bboxes_3d'].corners
                        bboxes_centers = results['gt_bboxes_3d'].center

                        image_aug_rgb = self.object_motion_sim_frontback(
                            image=img_rgb_255_np_uint8,
                            bboxes_centers=bboxes_centers,
                            bboxes_corners=bboxes_corners,
                            lidar2img=lidar2img,
                            # watch_img=True,
                            # file_path='2.jpg'
                        )
                        image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                        results['img'] = image_aug_bgr

                else:
                    # mono 数据集 nus kitti都只有一张图
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb = self.object_motion_sim_frontback_mono(
                        image=img_rgb_255_np_uint8,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        cam2img=cam2img,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr

        #######################################################
        # bbox operation
        #######################################################
        if 'bbox_shear' in self.corruption_severity_dict:

            # 双数据集  公共部分
            img_bgr_255_np_uint8 = results['img']  # nus:各论各的list / kitti: nparray

            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus:各论各的list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono 数据集
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('zzj: no lidar2img or cam_intrinsic found!')

            # points_tensor = results['points'].tensor
            bboxes_corners = results['gt_bboxes_3d'].corners
            bboxes_centers = results['gt_bboxes_3d'].center

            if type(bboxes_corners) == int:
                print(0)

            if type(bboxes_corners) != int:
            
                import numpy as np
                # 变换矩阵（和彩新代码统一）
                c = [0.05, 0.1, 0.15, 0.2, 0.25][self.bbox_shear.severity - 1]
                b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                transform_matrix = torch.tensor([
                    [1, 0, b],
                    [d, 1, e],
                    [f, 0, 1]
                ]).float()

                if not use_mono_dataset:

                    # nus 和 kitti 不同部分
                    if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(6):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_shear(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.png'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr
                    
                    elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(5):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_shear(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.png'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr

                    else:
                        # 判断是 kitti 数据集
                        img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                        image_aug_rgb = self.bbox_shear(
                            image=img_rgb_255_np_uint8,
                            bboxes_centers=bboxes_centers,
                            bboxes_corners=bboxes_corners,
                            transform_matrix=transform_matrix,
                            lidar2img=lidar2img,
                            # watch_img=True,
                            # file_path='2.png'
                        )
                        image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                        results['img'] = image_aug_bgr
                else:
                    # mono 数据集 nus kitti都只有一张图
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb = self.bbox_shear_mono(
                        image=img_rgb_255_np_uint8,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        transform_matrix=transform_matrix,
                        cam2img=cam2img,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr
                    # cv2.imwrite('img_log/2-out.jpg',image_aug_rgb[:,:,[2,1,0]])

                #############################################

        if 'bbox_rotate' in self.corruption_severity_dict:

            # 双数据集  公共部分
            img_bgr_255_np_uint8 = results['img']  # nus:各论各的list / kitti: nparray

            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus:各论各的list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono 数据集
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('zzj: no lidar2img or cam_intrinsic found!')

            #kcx 避免为空：
            bboxes_corners = results['gt_bboxes_3d'].corners
            bboxes_centers = results['gt_bboxes_3d'].center


            if type(bboxes_corners) == int:
                print(0)

            if type(bboxes_corners) != int:

                # 和彩新代码统一：
                # 仅绕z轴旋转
                # 疑问：是转20度左右？ 还是转20度以内随机？

                # theta_max = [4, 8, 12, 16, 20][self.bbox_rotate.severity-1]
                # theta = np.random.uniform(-theta_max, theta_max)
                import numpy as np
                theta_base = [4, 8, 12, 16, 20][self.bbox_rotate.severity - 1]
                theta_degree = np.random.uniform(theta_base - 2, theta_base + 2) * np.random.choice([-1, 1])

                theta = theta_degree / 180 * np.pi
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                if not use_mono_dataset:
                    # 非mono数据集，绕z轴旋转
                    transform_matrix = torch.tensor([
                        [cos_theta, sin_theta, 0],
                        [-sin_theta, cos_theta, 0],
                        [0, 0, 1],
                    ]).float()
                else:
                    # mono数据集，绕y轴旋转
                    transform_matrix = torch.tensor([
                        [cos_theta, 0, sin_theta],
                        [0, 1, 0],
                        [-sin_theta, 0, cos_theta],
                    ]).float()

                if not use_mono_dataset:

                    # nus 和 kitti 不同部分
                    if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(6):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_rotate(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.png'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr
                    
                    elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(5):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_rotate(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.png'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr

                    else:
                        # 判断是 kitti 数据集
                        img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                        image_aug_rgb = self.bbox_rotate(
                            image=img_rgb_255_np_uint8,
                            bboxes_centers=bboxes_centers,
                            bboxes_corners=bboxes_corners,
                            transform_matrix=transform_matrix,
                            lidar2img=lidar2img,
                            # watch_img=True,
                            # file_path='2.png'
                        )
                        image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                        results['img'] = image_aug_bgr
                else:
                    # mono 数据集 nus kitti都只有一张图
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb = self.bbox_rotate_mono(
                        image=img_rgb_255_np_uint8,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        transform_matrix=transform_matrix,
                        cam2img=cam2img,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr
                    # cv2.imwrite('img_log/2-out.jpg',image_aug_rgb[:,:,[2,1,0]])

            #############################################

        if 'bbox_scale' in self.corruption_severity_dict:
            # 双数据集  公共部分
            img_bgr_255_np_uint8 = results['img']  # nus:各论各的list / kitti: nparray

            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus:各论各的list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono 数据集
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('zzj: no lidar2img or cam_intrinsic found!')

            bboxes_corners = results['gt_bboxes_3d'].corners
            bboxes_centers = results['gt_bboxes_3d'].center

            if type(bboxes_corners) == int:
                print(0)

            if type(bboxes_corners) != int:
                # 尺缩，还能维度不统一的尺缩？xyz不一样的？有实际意义吗？
                c = [0.1, 0.2, 0.3, 0.4, 0.5][self.bbox_scale.severity - 1]
                a = b = d = 1
                import numpy as np
                r = np.random.randint(0, 3)
                t = np.random.choice([-1, 1])
            

                a += c * t
                b += c * t
                d += c * t
                # if r == 0:
                #     a += c * t     # 1 +- 0.1 +-0.2 ...
                #     b += c * (-t)
                # elif r == 1:
                #     b += c * t
                #     d += c * (-t)
                # elif r == 2:
                #     a += c * t
                #     d += c * (-t)

                transform_matrix = torch.tensor([
                    [a, 0, 0],
                    [0, b, 0],
                    [0, 0, d],
                ]).float()

                if not use_mono_dataset:

                    # nus 和 kitti 不同部分
                    if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(6):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_scale(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.jpg'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr

                    elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                        # 判断是否是 nus 数据集
                        '''
                        nuscenes:
                        0    CAM_FRONT,
                        1    CAM_FRONT_RIGHT,
                        2    CAM_FRONT_LEFT,
                        3    CAM_BACK,
                        4    CAM_BACK_LEFT,
                        5    CAM_BACK_RIGHT
                        '''
                        image_aug_bgr = []
                        for i in range(5):
                            img_rgb_255_np_uint8_i = img_bgr_255_np_uint8[i][:, :, [2, 1, 0]]
                            lidar2img_i = lidar2img[i]
                            image_aug_rgb_i = self.bbox_scale(
                                image=img_rgb_255_np_uint8_i,
                                bboxes_centers=bboxes_centers,
                                bboxes_corners=bboxes_corners,
                                transform_matrix=transform_matrix,
                                lidar2img=lidar2img_i,
                                is_nus=True,
                                # watch_img=True,
                                # file_path='img_log/'+str(i)+'-watch.jpg'
                            )
                            image_aug_bgr_i = image_aug_rgb_i[:, :, [2, 1, 0]]
                            image_aug_bgr.append(image_aug_bgr_i)
                            # cv2.imwrite('img_log/'+str(i)+'-in.png', img_rgb_255_np_uint8_i[:,:,[2,1,0]])
                            # cv2.imwrite('img_log/'+str(i)+'-out.png',image_aug_rgb_i[:,:,[2,1,0]])
                        results['img'] = image_aug_bgr

                    else:
                        # 判断是 kitti 数据集
                        img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                        image_aug_rgb = self.bbox_scale(
                            image=img_rgb_255_np_uint8,
                            bboxes_centers=bboxes_centers,
                            bboxes_corners=bboxes_corners,
                            transform_matrix=transform_matrix,
                            lidar2img=lidar2img,
                            # watch_img=True,
                            # file_path='2.jpg'
                        )
                        image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                        results['img'] = image_aug_bgr

                else:
                    # mono 数据集 nus kitti都只有一张图
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb = self.bbox_scale_mono(
                        image=img_rgb_255_np_uint8,
                        bboxes_centers=bboxes_centers,
                        bboxes_corners=bboxes_corners,
                        transform_matrix=transform_matrix,
                        cam2img=cam2img,
                        # watch_img=True,
                        # file_path='2.jpg'
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr
                    # cv2.imwrite('img_log/2-out.jpg',image_aug_rgb[:,:,[2,1,0]])

    #############################################

        #lidar corruptions

        if 'gaussian_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['gaussian_noise']
           # aug_pl = pl[:,:3]
            points_aug = gaussian_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'lidar_crosstalk_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['lidar_crosstalk_noise']
            aug_pl = pl[:,:3]
           # aug_pl = pl[:,:3]
            points_aug = lidar_crosstalk_noise(pl.numpy(), severity)
            np.save('/data/home/scv7306/run/aaa/TransFusion2/lidar.npz',points_aug)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'density_dec_global' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['density_dec_global']
            # aug_pl = pl[:,:3]
            points_aug = density_dec_global(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'density_dec_local' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['density_dec_local']
            # aug_pl = pl[:,:3]
            points_aug = density_dec_local(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        
        if 'cutout_local' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['cutout_local']
            # aug_pl = pl[:,:3]
            points_aug = cutout_local(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'scene_glare_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['scene_glare_noise']
            # aug_pl = pl[:,:3]
            points_aug = scene_glare_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'uniform_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['uniform_noise']
            # aug_pl = pl[:,:3]
            points_aug = uniform_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'upsampling' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['upsampling']
            # aug_pl = pl[:,:3]
            points_aug = upsampling(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'background_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['background_noise']
           # aug_pl = pl[:,:3]
            points_aug = background_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl
        
        if 'impulse_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['impulse_noise']
           # aug_pl = pl[:,:3]
            points_aug = impulse_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'layer_del' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['layer_del']
            # aug_pl = pl[:,:3]
            points_aug = layer_del(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'rain_sim_lidar' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['rain_sim_lidar']
            # aug_pl = pl[:,:3]
            points_aug = rain_sim_lidar(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'fog_sim_lidar' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['fog_sim_lidar']
            # aug_pl = pl[:,:3]
            points_aug = fog_sim_lidar(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'snow_sim_lidar' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['snow_sim_lidar']
            # aug_pl = pl[:,:3]
            points_aug = snow_sim_lidar(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'spatial_alignment_noise' in self.corruption_severity_dict:
            ori_pose = results['lidar2img']
            # print(ori_pose)
            # print(type(ori_pose))
            import numpy as np
            temp = np.array(ori_pose)
            # print(temp.shape)
            severity = self.corruption_severity_dict['spatial_alignment_noise']
            # aug_pl = pl[:,:3]
            noise_pose = ori_pose
            if len(ori_pose) == 5:
                for i in range(5):
                    noise_pose[i] = spatial_alignment_noise(ori_pose[i], severity)

            else:
                noise_pose = spatial_alignment_noise(ori_pose, severity)
            # print(noise_pose)
            # pl = torch.from_numpy(points_aug)
            # results['points'].tensor = pl
            results['lidar2img'] = noise_pose
            # print(results['lidar2img'])

        # if 'temporal_alignment_noise' in self.corruption_severity_dict:
        #     img_bgr_255_np_uint8 = results['img']
        #     print('---'*10)
        #     severity = self.corruption_severity_dict['temporal_alignment_noise']
        #     frame = temporal_alignment_noise(severity)

        #     ## 替换图片
        #     if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
        #         '''
        #         nuscenes:
        #         0    CAM_FRONT,
        #         1    CAM_FRONT_RIGHT,
        #         2    CAM_FRONT_LEFT,
        #         3    CAM_BACK,
        #         4    CAM_BACK_LEFT,
        #         5    CAM_BACK_RIGHT
        #         '''
        #         image_aug_bgr = []
        #         for i in range(6):
        #             filename = results['cam_sweeps'][i][frame-1]['filename']
        #             img = mmcv.imread('/data/home/scv7306/run/aaa/TransFusion2/data/nuscenes'+filename)
        #             # img_rgb_255_np_uint8_i = img
        #             image_aug_bgr_i = img
        #             image_aug_bgr.append(image_aug_bgr_i)
        #         results['img'] = image_aug_bgr
            
        #     ## 替换lidar
        #     lidar_info = results['sweeps']
        #     sweep = lidar_info[frame-1]
        #     points_sweep = load_points(sweep['data_path'])
        #     points_sweep = np.copy(points_sweep).reshape(-1, 5)
        #     results['points'].tensor = torch.from_numpy(points_sweep)
        
        if 'temporal_alignment_noise' in self.corruption_severity_dict:
            import numpy as np
            img_bgr_255_np_uint8 = results['img']
            # print('---'*10)
            severity = self.corruption_severity_dict['temporal_alignment_noise']
            frame = temporal_alignment_noise(severity) 

            ## 替换图片
            # print(len(results['cam_sweeps'][1]))
            # print(results['cam_sweeps'])
            if len(results['cam_sweeps'][0]) == 0:
                assss = 1
            else:
                if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                    '''
                    nuscenes:
                    0    CAM_FRONT,
                    1    CAM_FRONT_RIGHT,
                    2    CAM_FRONT_LEFT,
                    3    CAM_BACK,
                    4    CAM_BACK_LEFT,
                    5    CAM_BACK_RIGHT
                    '''
                    image_aug_bgr = []
                    for i in range(6):
                        # filename = '/data/public/nuscenes/'+results['cam_sweeps'][6-i-1][frame-1]['filename']
                        filename = '/data/public/nuscenes/'+results['cam_sweeps'][i][frame-1]['filename']
                        img = mmcv.imread(filename)
                        image_aug_bgr.append(img)
                    # print('ori_img',results['img'])
                    # print('---'*10)
                    # print('changed_img',image_aug_bgr)
                    results['img'] = image_aug_bgr
            
            ## 替换lidar
            lidar_info = results['sweeps']
            # print('len of lidar', len(lidar_info))
            if len(lidar_info) < frame-1:
                assss = 1
            else:
                while (len(lidar_info) <= frame-1):
                    frame = frame-1
                    # print(frame)
                sweep = lidar_info[frame-1]
                points_sweep = load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, 5)
                points_sweep = remove_close(points_sweep)
                # print(points_sweep.shape)
                results['points'].tensor = torch.from_numpy(points_sweep)
                # results['points'].tensor[:10000,:] = torch.from_numpy(points_sweep)[:10000,:] #[:,[0, 1, 2, 4]]

        if 'fulltrajectory_noise' in self.corruption_severity_dict:
            import numpy as np
            # from .generate_lidar_c_new import fulltrajectory_noise
            pl = results['points'].tensor
            # print(pl.size())
            #print(results)
            severity = self.corruption_severity_dict['fulltrajectory_noise']
            aug_pl = pl[:,:3]
            # load pc_pose
            if len(results['sweeps_back']) != 0:
            
                fir_ego_pose = format_list_float_06(results['ego2global_translation'] + results['ego2global_rotation'])
                fir_sen_glo = format_list_float_06(results['lidar2ego_translation'] + results['lidar2ego_rotation'])

                sec_sweeps = results['sweeps_back'][0]
                sec_ego_pose = format_list_float_06(sec_sweeps['ego2global_translation']+ sec_sweeps['ego2global_rotation'])
                sec_sen_glo = format_list_float_06(sec_sweeps['sensor2ego_translation']+ sec_sweeps['sensor2ego_rotation'])

                pc_pose = np.array([fir_ego_pose,fir_sen_glo,sec_ego_pose,sec_sen_glo])
                # print(pc_pose.shape)
                points_aug = fulltrajectory_noise(aug_pl.numpy(), pc_pose, severity)
                pl[:,:3] = points_aug
                results['points'].tensor = pl
            else:
                results['points'].tensor = pl

        
        if 'fov' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['fov']
            # aug_pl = pl[:,:3]
            points_aug = filter_point_by_angle(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl





            


# bbox
        
        if 'cutout_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            # data.append(results['gt_bboxes_3d'])
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['cutout_bbox']
            # aug_pl = pl[:,:3]
            points_aug = cutout_bbox(pl.numpy(), severity,data)
            # points_aug = cutout_bbox(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'density_dec_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['density_dec_bbox']
            # aug_pl = pl[:,:3]
            points_aug = density_dec_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'gaussian_noise_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['gaussian_noise_bbox']
            # aug_pl = pl[:,:3]
            points_aug = gaussian_noise_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'scale_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['scale_bbox']
            aug_pl = pl[:,:4]
            points_aug = scale_bbox(aug_pl.numpy(), severity,data)
            pl[:,:4] = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'shear_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['shear_bbox']
            aug_pl = pl[:,:4]
            points_aug = shear_bbox(aug_pl.numpy(), severity,data)
            pl[:,:4] = torch.from_numpy(points_aug)
            results['points'].tensor = pl


        if 'FFD_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['FFD_bbox']
            # aug_pl = pl[:,:3]
            points_aug = FFD_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'moving_noise_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['moving_noise_bbox']
            # aug_pl = pl[:,:3]
            points_aug = moving_noise_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'uniform_noise_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['uniform_noise_bbox']
            # aug_pl = pl[:,:3]
            points_aug = uniform_noise_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'upsampling_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['upsampling_bbox']
            # aug_pl = pl[:,:3]
            points_aug = upsampling_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'impulse_noise_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['impulse_noise_bbox']
            # aug_pl = pl[:,:3]
            points_aug = impulse_noise_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        if 'rotation_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #适配waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            # data.append(results['lidar2ego_rotation'])
            # data.append(results['lidar2ego_translation'])
            # data.append(results['ego2global_rotation'])
            # data.append(results['ego2global_translation'])
            # print(data)
            severity = self.corruption_severity_dict['rotation_bbox']
            aug_pl = pl[:,:4]
            points_aug = rotation_bbox(aug_pl.numpy(), severity,data)
            pl[:,:4] = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
