 # Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving

Accepted to CVPR2023. 🔥


Here we only provide the functions of all corruptions in 3D object detection. The whole project is built upon MMDetection3D and OpenPCDet with necessary modifications of its source code.

## Prerequisites
* Python (3.8.2)
* Pytorch (1.9.0)
* numpy
* imagecorruptions


## How to use?

### Select severity
The severity can be selected from [1,2,3,4,5]

### LiDAR corruptions

* LiDAR corruptions and corresponding function

|  corruptions   | function  |
|  :----:  | :----:  |
|  Snow   |  ``snow_sim``/``snow_sim_nus``  |
|  Rain   | ``rain_sim`` |
|  Fog   | ``fog_sim``  |
|  Strong Sunlight   | ``scene_glare_noise``  |
|  Density Decrease   | ``density_dec_global``  |
|  Cutout   | ``cutout_local``  |
|  LiDAR Crosstalk   | ``lidar_crosstalk_noise``  |
|  FOV Lost   | ``fov_filter``  |
|  Gaussian Noise   | ``gaussian_noise``  |
|  Uniform Noise   | ``uniform_noise``  |
|  Impulse Noise   | ``impulse_noise``  |
|  Motion Compensation   | ``fulltrajectory_noise``  |
|  Moving Object   | ``moving_noise_bbox``  |
|  Local Density Decrease   | ``density_dec_bbox``  |
|  Local Cutout   | ``cutout_bbox``  |
|  Local Gaussian Noise   | ``gaussian_noise_bbox``  |
|  Local Uniform Noise   | ``uniform_noise_bbox``  |
|  Local Impulse Noise   | ``impulse_noise_bbox``  |
|  Shear   | ``shear_bbox``  |
|  Scale   | ``scale_bbox``  |
|  Rotation   | ``rotation_bbox``  |
|  Spatial Misalignment   | ``spatial_alignment_noise``  |
|  Temporal Misalignment   | ``temporal_alignment_noise``  |

```python
# first, make sure the point cloud in a numpy array format, like N*4 or N*5
lidar = np.array([N,4])

# weather-level
from .LiDAR_corruptions import rain_sim, snow_sim, fog_sim

lidar_cor = rain_sim(lidar, severity)

# other corruption can be the same operation, like gaussian_noise,lidar_crosstalk_noise,density_dec_global,density_dec_local,cutout_local,uniform_noise

# Note that the object-level corruption need 3D bounding box information as input, like results['ann_info']['gt_bboxes_3d'] in mmdet3d

from .LiDAR_corruptions import gaussian_noise_bbox
bbox = results['ann_info']['gt_bboxes_3d']
lidar_cor = gaussian_noise_bbox(lidar, severity,bbox)


# Moreover, the temporal_alignment_noise needs ego pose information, like results['lidar2img'] in mmdet3d
from .LiDAR_corruptions import temporal_alignment_noise

noise_pose = spatial_alignment_noise(ori_pose, severity)



```


### Camera corruptions


* Camera corruptions and corresponding function

|  corruptions   | function  |
|  :----:  | :----:  |
|  Snow   | ``ImageAddSnow``  |
|  Rain   | ``ImageAddRain``  |
|  Fog   | ``ImageAddFog``  |
|  Strong Sunlight   | ``ImagePointAddSun``  |
|  Gaussian Noise   | ``ImageAddGaussianNoise``  |
|  Uniform Noise   | ``ImageAddUniformNoise``  |
|  Impulse Noise   | ``ImageAddImpulseNoise``  |
|  Moving Object   | ``ImageBBoxMotionBlurFrontBack``/``ImageBBoxMotionBlurLeftRight``  |
|  Motion Blur   | ``ImageMotionBlurFrontBack``/``ImageMotionBlurLeftRight``  |
|  Shear   | ``ImageBBoxOperation``  |
|  Scale   | ``ImageBBoxOperation``  |
|  Rotation   | ``ImageBBoxOperation``  |
|  Spatial Misalignment   | ``spatial_alignment_noise``  |
|  Temporal Misalignment   | ``temporal_alignment_noise``  |

```python
# weather-level
from .Camera_corruptions import ImageAddSnow,ImageAddFog,ImageAddRain
snow_sim = ImageAddSnow(severity, seed=2022)
img_bgr_255_np_uint8 = results['img'] # the img in mmdet3d loading pipeline
img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:,:,[2,1,0]]
image_aug_rgb = snow_sim(
    image=img_rgb_255_np_uint8
    )
image_aug_bgr = image_aug_rgb[:,:,[2,1,0]]
results['img'] = image_aug_bgr


# other corruption can be the same operation, like ImageAddGaussianNoise,ImageAddImpulseNoise,ImageAddUniformNoise

# Note that the object-level corruption need 3D bounding box information as input, like results['gt_bboxes_3d'] in mmdet3d
from .Camera_corruptions import ImageBBoxOperation
bbox_shear = ImageBBoxOperation(severity)
img_bgr_255_np_uint8 = results['img']
bboxes_corners = results['gt_bboxes_3d'].corners
bboxes_centers = results['gt_bboxes_3d'].center
lidar2img = results['lidar2img'] 
img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
c = [0.05, 0.1, 0.15, 0.2, 0.25][bbox_shear.severity - 1]
b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
transform_matrix = torch.tensor([
    [1, 0, b],
    [d, 1, e],
    [f, 0, 1]
]).float()

image_aug_rgb = bbox_shear(
    image=img_rgb_255_np_uint8,
    bboxes_centers=bboxes_centers,
    bboxes_corners=bboxes_corners,
    transform_matrix=transform_matrix,
    lidar2img=lidar2img,
)
image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
results['img'] = image_aug_bgr

```


### Examples within MMDetection3D pipeline

In the MMDetection3D pipeline, we modified the `mmdetection3d/mmdet3d/datasets/pipelines/test_time_aug.py`, the modification examples are:
```python
@PIPELINES.register_module()
class CorruptionMethods(object):
    """Test-time augmentation with corruptions.

    """

    def __init__(self,
                 corruption_severity_dict=
                    {
                        'sun_sim':2,
                    },
                 ):
                 
        self.corruption_severity_dict = corruption_severity_dict
        
        if 'sun_sim' in self.corruption_severity_dict:
            np.random.seed(2022)
            severity = self.corruption_severity_dict['sun_sim']
            self.sun_sim = ImagePointAddSun(severity)
            self.sun_sim_mono = ImageAddSunMono(severity)
            
         #for multiple cameras corruption
         
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
            
            
    
def __call__(self, results):
        """Call function to augment common corruptions.
        """
        if 'sun_sim' in self.corruption_severity_dict:
            # Common part of two datasets
            img_bgr_255_np_uint8 = results['img']  # nus:list / kitti: nparray

            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  # nus: list / kitti: nparray
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  # for xxx-mono dataset
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('no lidar2img or cam_intrinsic found!')

            if not use_mono_dataset:
                points_tensor = results['points'].tensor
                # different part of nus and kitti
                if type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 6:
                    # nus dataset
                    # only one sun
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
                    )
                    image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
                    img_bgr_255_np_uint8[0] = image_aug_bgr_0
                    results['img'] = img_bgr_255_np_uint8
                    results['points'].tensor = points_aug

                elif type(img_bgr_255_np_uint8) == list and len(img_bgr_255_np_uint8) == 5:
                    # waymo dataset
                    # only one sun
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
                    )
                    image_aug_bgr_0 = image_aug_rgb_0[:, :, [2, 1, 0]]
                    img_bgr_255_np_uint8[0] = image_aug_bgr_0
                    results['img'] = img_bgr_255_np_uint8
                    results['points'].tensor = points_aug

                else:
                    # kitti dataset
                    img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                    image_aug_rgb, points_aug = self.sun_sim(
                        image=img_rgb_255_np_uint8,
                        points=points_tensor,
                        lidar2img=lidar2img,
                    )
                    image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                    results['img'] = image_aug_bgr
                    results['points'].tensor = points_aug
            else:
                # mono dataset of nus and kitti only one image
                img_rgb_255_np_uint8 = img_bgr_255_np_uint8[:, :, [2, 1, 0]]
                image_aug_rgb = self.sun_sim_mono(
                    image=img_rgb_255_np_uint8,
                )
                image_aug_bgr = image_aug_rgb[:, :, [2, 1, 0]]
                results['img'] = image_aug_bgr

        if 'object_motion_sim' in self.corruption_severity_dict:
            img_bgr_255_np_uint8 = results['img']
            # points_tensor = results['points'].tensor
            if 'lidar2img' in results:
                lidar2img = results['lidar2img']  
                use_mono_dataset = False
            elif 'cam_intrinsic' in results['img_info']:
                cam2img = results['img_info']['cam_intrinsic']  
                import numpy as np
                cam2img = np.array(cam2img)
                use_mono_dataset = True
            else:
                raise AssertionError('no lidar2img or cam_intrinsic found!')
            
            bboxes_corners = results['gt_bboxes_3d'].corners
            bboxes_centers = results['gt_bboxes_3d'].center


            if type(bboxes_corners) == int:
                print(0)

            if type(bboxes_corners) != int:

                if not use_mono_dataset:
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


        
        #for lidar corruptions

        if 'gaussian_noise' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            severity = self.corruption_severity_dict['gaussian_noise']
           # aug_pl = pl[:,:3]
            points_aug = gaussian_noise(pl.numpy(), severity)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl

        #for lidar corruptions with bbox
        
        if 'cutout_bbox' in self.corruption_severity_dict:
            import numpy as np
            pl = results['points'].tensor
            data = []
            # data.append(results['gt_bboxes_3d'])
            if 'gt_bboxes_3d' in results:
                data.append(results['gt_bboxes_3d'])
            else:
                #waymo
                data.append(results['ann_info']['gt_bboxes_3d'])
            severity = self.corruption_severity_dict['cutout_bbox']
            points_aug = cutout_bbox(pl.numpy(), severity,data)
            pl = torch.from_numpy(points_aug)
            results['points'].tensor = pl


                
            


```

Then add 'CorruptionMethods dict' to the test pipeline:
modify corresponding config file in `mmdetection3d/configs/`, the modification examples are:

```python
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='CorruptionMethods',
        corruption_severity_dict=
            {
                'background_noise':1,
                # 'snow':2,
                # 'gaussian_noise_points':2,
            },
    ),
    ...]

```





### Examples within OpenPCDet pipeline

```python
class KittiDataset(DatasetTemplate):
    # def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None ):
    def __init__(self, dataset_cfg, class_names, corruptions, severity, training=True, root_path=None, logger=None ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
    
    # for lidar 
    def get_lidar(self, idx):
        try:
            if "bbox" in self.corruptions[0]:
                data = MAP[self.corruptions[0]](data,self.severity,idx)
            else:
                data = MAP[self.corruptions[0]](data,self.severity)
        except:
            pass
        
        return data


    # for image
    def get_image(self, idx):
        if self.corruptions[1] == 'rain_sim':
            image_add_some_func = ImageAddRain(severity=self.severity, seed=2022)
            image = image_add_some_func(image, True,'./test.png')
        





```


