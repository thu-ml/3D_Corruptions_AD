import sys
sys.path.append('utils')


import imgaug.augmenters as iaa
import weather.Automold as am
from torchvision.utils import save_image
import torch
import cv2
import numpy as np
from scipy.ndimage import zoom as scizoom

import torch.nn.functional as F 
from utils.tps_grid_gen import TPSGridGen

#YOU NEED : pip install imagecorruptions


import scipy.stats as st

def get_gaussian_kernel(kernlen=5, nsig=3):  
    interval = (2*nsig+1.)/kernlen   
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1) 
    kern1d = np.diff(st.norm.cdf(x))  
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))  
    kernel = kernel_raw/kernel_raw.sum()            
    return kernel



from mmdet3d.core.bbox.structures.utils import points_cam2img


class ImagePointAddSun():
    def __init__(self, severity) -> None:
        print(self.__class__.__name__, ': please set numpy seed !')
        self.severity = severity

    def __call__(self, image, points, lidar2img, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB

            points should be tensor : N * 4
            
        """
        severity = self.severity
        temp_dict_trans_information = {}

        image_aug = self.sun_sim_img(image, lidar2img, severity, watch_img, file_path, temp_dict_trans_information)
        points_aug = self.sun_sim_point(points, lidar2img, severity, watch_img, file_path, temp_dict_trans_information)


        return image_aug, points_aug
    


    def sun_sim_img(self, image, lidar2img, severity, watch_img=False, file_path='', temp_dict_trans_information={}):
        sun_radius = [30, 40, 50, 60, 70][severity-1]
        
        # corruption severity-independent parameters
        no_of_flare_circles = 1
        src_color = (255,255,255)

        img_width = image.shape[1]
        img_height = image.shape[0]
        sun_u_range = [0.25, 0.75]
        sun_v_range = [0.30, 0.45]
        sun_u = np.random.uniform(*sun_u_range)*img_width
        sun_v = np.random.uniform(*sun_v_range)*img_height
        sun_uv = np.array([sun_u, sun_v])


        img_center_coor = np.array([image.shape[1]/2, image.shape[0]/2])
        sun_uv_to_center = img_center_coor - sun_uv

        sun_flare_line_angle = np.arctan(sun_uv_to_center[1]/sun_uv_to_center[0])

        image_uint8_rgb = image

        flare_image_rgb = image.astype(np.float64)
        mainflare_mask = np.zeros_like(image).astype(np.float64)
        try:
            flare_image_rgb, mainflare_mask = am.add_sun_flare(
                image_uint8_rgb, 
                flare_center=sun_uv, 
                angle=sun_flare_line_angle,
                no_of_flare_circles = no_of_flare_circles,
                src_radius=sun_radius,
                src_color=src_color
                )
        except:
            pass

        flare_image_rgb_uint8 = flare_image_rgb.copy().astype(np.uint8)
        
        #################################################################
        # split  image  and  point
        #################################################################
        temp_dict_trans_information['sun_sim'] = {}
        temp_dict_trans_information['sun_sim']['mainflare_mask'] = mainflare_mask
        temp_dict_trans_information['sun_sim']['sun_uv'] = sun_uv

        if watch_img:
            flare_image_save = torch.from_numpy(flare_image_rgb).permute(2,0,1).float() /255.
            temp_dict_trans_information['sun_sim']['flare_image_save'] = flare_image_save

        return flare_image_rgb_uint8




    def sun_sim_point(self, points, lidar2img, severity, watch_img=False, file_path='', temp_dict_trans_information={}):
        # get data
        # points = results['points'].tensor
        # lidar2img = results['lidar2img']
        mainflare_mask = temp_dict_trans_information['sun_sim']['mainflare_mask']
        sun_uv = temp_dict_trans_information['sun_sim']['sun_uv']

        noisy_point_ratio = [0.004, 0.008, 0.012, 0.016, 0.020][severity-1]

        if not (mainflare_mask==0).all():
            # remove the points in sun use a radias
            mask_ther = 0.8
            mainflare_mask_lidar = (mainflare_mask[...,0] > (mask_ther*255.)).astype(np.uint8)
            sun_radis_nonzero = mainflare_mask_lidar.sum(0).nonzero()[0]
            sun_radis = (sun_radis_nonzero.max() - sun_radis_nonzero.min()) / 2
            pts_uvd = points_cam2img(points[:,:3], proj_mat=lidar2img, with_depth=True)
            pts_2d = pts_uvd[:,:2]
            pts_depth = pts_uvd[:,2]
            pts_keep_flag = ((pts_2d - sun_uv[None]).pow(2).sum(1).sqrt() > sun_radis) + (pts_depth<0)
            points_keep = points[pts_keep_flag]

        else:
            mask_ther = 0.8
            mainflare_mask_lidar = (mainflare_mask[...,0] > (mask_ther*255.)).astype(np.uint8)
            pts_uvd = points_cam2img(points[:,:3], proj_mat=lidar2img, with_depth=True)
            pts_2d = pts_uvd[:,:2]
            pts_depth = pts_uvd[:,2]
            points_keep = points


        if watch_img:
            mainflare_mask_lidar_point = np.repeat(mainflare_mask_lidar[:,:,None], 3, axis=2) * 255
            mainflare_mask_lidar_point = mainflare_mask_lidar_point.astype(np.uint8)

            front_pts_2d = pts_2d[pts_depth>0]
            for coor in front_pts_2d:
                point_size = 1
                point_color = (255,0,0)
                thickness = 1
                cv2.circle(mainflare_mask_lidar_point, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)

            mainflare_mask_lidar_point_save = torch.from_numpy(mainflare_mask_lidar_point).permute(2,0,1).float() /255.
        ################################
        points_keep_with_noise = self._lidar_sun_noise(points_keep, noisy_point_ratio)
        if watch_img:
            flare_image_save = temp_dict_trans_information['sun_sim']['flare_image_save']

            pts_uvd_noise = points_cam2img(points_keep_with_noise[:,:3], proj_mat=lidar2img, with_depth=True)
            front_point_noise = pts_uvd_noise[pts_uvd_noise[:,2]>0]
            pts_2d_noise = front_point_noise[:,:2]
            mainflare_mask_lidar_point_noise = np.repeat(mainflare_mask_lidar[:,:,None], 3, axis=2) * 255
            mainflare_mask_lidar_point_noise = mainflare_mask_lidar_point_noise.astype(np.uint8)

            for coor in pts_2d_noise:
                point_size = 1
                point_color = (255,0,0)
                thickness = 1
                cv2.circle(mainflare_mask_lidar_point_noise, (int(coor[0]),int(coor[1])), point_size, point_color, thickness)

            mainflare_mask_lidar_point_noise_save = torch.from_numpy(mainflare_mask_lidar_point_noise).permute(2,0,1).float() /255.
            save_image([flare_image_save, mainflare_mask_lidar_point_save, mainflare_mask_lidar_point_noise_save], padding=0, pad_value=255., nrow=1, fp=file_path)


        # put back to results
        return points_keep_with_noise


    def _lidar_sun_noise(self, pointcloud, noisy_point_ratio):
        N, C = pointcloud.shape
        # x y z i
        xyz_channel = 3
        noisy_point_num = int(N * noisy_point_ratio)
        index = np.random.choice(N, noisy_point_num, replace=False) 
        pointcloud[index, :xyz_channel] += torch.randn(noisy_point_num, xyz_channel) * 3.0 
        return pointcloud







class ImageAddSunMono():
    def __init__(self, severity) -> None:
        print(self.__class__.__name__, ': please set numpy seed !')
        self.severity = severity

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB

        """
        severity = self.severity
        temp_dict_trans_information = {}

        image_aug = self.sun_sim_img(image, severity, watch_img, file_path, temp_dict_trans_information)

        return image_aug
    


    def sun_sim_img(self, image, severity, watch_img=False, file_path='', temp_dict_trans_information={}):
        # get data
        # image must be img_rgb_255_np_uint8
        

        # corruption severity-related parameters
        sun_radius = [30, 40, 50, 60, 70][severity-1]
        
        # corruption severity-independent parameters
        

        no_of_flare_circles = 1
        src_color = (255,255,255)

        img_width = image.shape[1]
        img_height = image.shape[0]
        sun_u_range = [0.25, 0.75]
        sun_v_range = [0.30, 0.45]
        sun_u = np.random.uniform(*sun_u_range)*img_width
        sun_v = np.random.uniform(*sun_v_range)*img_height
        sun_uv = np.array([sun_u, sun_v])

        img_center_coor = np.array([image.shape[1]/2, image.shape[0]/2])
        sun_uv_to_center = img_center_coor - sun_uv

        sun_flare_line_angle = np.arctan(sun_uv_to_center[1]/sun_uv_to_center[0])
        sun_flare_line_angle_degree = sun_flare_line_angle/np.pi*180

        image_uint8_rgb = image
        flare_image_rgb, mainflare_mask = am.add_sun_flare(
            image_uint8_rgb, 
            flare_center=sun_uv, 
            angle=sun_flare_line_angle,
            no_of_flare_circles = no_of_flare_circles,
            src_radius=sun_radius,
            src_color=src_color
            )
        flare_image_rgb_uint8 = flare_image_rgb.copy().astype(np.uint8)
        
        #################################################################
        # split  image  and  point
        #################################################################
        temp_dict_trans_information['sun_sim'] = {}
        temp_dict_trans_information['sun_sim']['mainflare_mask'] = mainflare_mask
        temp_dict_trans_information['sun_sim']['sun_uv'] = sun_uv

        if watch_img:
            flare_image_save = torch.from_numpy(flare_image_rgb).permute(2,0,1).float() /255.
            save_image(flare_image_save, padding=0, pad_value=255., nrow=1, fp=file_path)


        return flare_image_rgb_uint8





import time

class ImageBBoxOperation(): 
    def __init__(self, severity) -> None:
        self.severity = severity

    def __call__(self, image, lidar2img, transform_matrix, 
                bboxes_centers, bboxes_corners, 
                watch_img=False, file_path='',
                is_nus=False,
                ) -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_tensor_org = torch.from_numpy(image).permute(2,0,1).float() / 255.
        image_tensor = image_tensor_org.clone()
        img_height=image_tensor.shape[1]
        img_width=image_tensor.shape[2]

        lidar2img = lidar2img.astype(np.float32)

        # bboxes_corners 3*8*3   tensor 
        # lidar2img 4*4   np.array

        # canvas = np.zeros((img_height, img_width,3)).astype(np.uint8)
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        affine_mat = torch.Tensor([
            [1,0,0],
            [0,1,0]
        ])

        bboxes_centers_distance_pow2 = bboxes_centers[:,:2].pow(2).sum(1)
        bboxes_centers_distance_index_far2near = bboxes_centers_distance_pow2.sort(descending=True)[1]

        bboxes_corners = bboxes_corners[bboxes_centers_distance_index_far2near]
        bboxes_centers = bboxes_centers[bboxes_centers_distance_index_far2near]

        imge_changed_flag = False
        for idx_b in range(bboxes_num):
            # time_last = time.time()


            corners = bboxes_corners[idx_b]   # 8*3
            bboxes_center = bboxes_centers[idx_b]


            continue_flag, imge_changed_flag, target_points, source_points, \
            right_line_index, left_line_index, mid_line_index = get_control_point(
                    corners, 
                    bboxes_center, 
                    lidar2img,
                    transform_matrix,
                    img_height,
                    img_width,
                    imge_changed_flag
                )
            if continue_flag:
                continue
            
            smaller_flag = False
            if transform_matrix[0,0] != 0 \
                and transform_matrix[1,1] != 0 \
                and transform_matrix[2,2] != 0 \
                and transform_matrix[0,1] == 0 \
                and transform_matrix[0,2] == 0 \
                and transform_matrix[1,0] == 0 \
                and transform_matrix[1,2] == 0 \
                and transform_matrix[2,0] == 0 \
                and transform_matrix[2,1] == 0:
                if transform_matrix[0,0] < 1 or transform_matrix[1,1] < 1 or transform_matrix[2,2] < 1:
                    smaller_flag = True


            target_image, canvas, imge_changed_flag = obj_img_transform(
                image_tensor,
                imge_changed_flag, target_points, source_points, 
                right_line_index, left_line_index, mid_line_index,
                watch_img, canvas,
                smaller_flag
            )

            # replace
            image_tensor = target_image
            

            


        if watch_img:
            canvas_tensor = torch.from_numpy(canvas).float().permute(2,0,1)/255
            if imge_changed_flag:
                save_image([canvas_tensor, target_image], nrow=1, fp=file_path)
            else:
                save_image([canvas_tensor, image_tensor], nrow=1, fp=file_path)
        ################  finish  then output  #############
        if imge_changed_flag:
            image_aug = target_image
            image_aug_np_rgb_255 = (image_aug*255).permute(1,2,0).numpy().astype(np.uint8)
            return image_aug_np_rgb_255
        else:
            # no change in image
            return image

        


class ImageBBoxOperationMono(): 
    def __init__(self, severity) -> None:
        self.severity = severity

    def __call__(self, image, cam2img, transform_matrix, 
                bboxes_centers, bboxes_corners, 
                watch_img=False, file_path='',
                is_nus=False,
                ) -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_tensor_org = torch.from_numpy(image).permute(2,0,1).float() / 255.
        image_tensor = image_tensor_org.clone()
        img_height=image_tensor.shape[1]
        img_width=image_tensor.shape[2]

        cam2img = cam2img.astype(np.float32)

        # bboxes_corners 3*8*3   tensor 
        # cam2img 4*4   np.array

        # canvas = np.zeros((img_height, img_width,3)).astype(np.uint8)
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        affine_mat = torch.Tensor([
            [1,0,0],
            [0,1,0]
        ])



        bboxes_centers_distance_pow2 = bboxes_centers[:,:2].pow(2).sum(1)
        bboxes_centers_distance_index_far2near = bboxes_centers_distance_pow2.sort(descending=True)[1]


        bboxes_corners = bboxes_corners[bboxes_centers_distance_index_far2near]
        bboxes_centers = bboxes_centers[bboxes_centers_distance_index_far2near]

        imge_changed_flag = False
        for idx_b in range(bboxes_num):
            # time_last = time.time()
            
            corners = bboxes_corners[idx_b]   # 8*3
            bboxes_center = bboxes_centers[idx_b]


            continue_flag, imge_changed_flag, target_points, source_points, \
            right_line_index, left_line_index, mid_line_index = get_control_point_mono(
                    corners, 
                    bboxes_center, 
                    cam2img,
                    transform_matrix,
                    img_height,
                    img_width,
                    imge_changed_flag
                )
            if continue_flag:
                continue


            # $==================================================================
            smaller_flag = False
            if transform_matrix[0,0] != 0 \
                and transform_matrix[1,1] != 0 \
                and transform_matrix[2,2] != 0 \
                and transform_matrix[0,1] == 0 \
                and transform_matrix[0,2] == 0 \
                and transform_matrix[1,0] == 0 \
                and transform_matrix[1,2] == 0 \
                and transform_matrix[2,0] == 0 \
                and transform_matrix[2,1] == 0:
                if transform_matrix[0,0] < 1 or transform_matrix[1,1] < 1 or transform_matrix[2,2] < 1:
                    smaller_flag = True

            target_image, canvas, imge_changed_flag = obj_img_transform(
                image_tensor,
                imge_changed_flag, target_points, source_points, 
                right_line_index, left_line_index, mid_line_index,
                watch_img, canvas,
                smaller_flag
            )

            # replace
            image_tensor = target_image


        if watch_img:
            canvas_tensor = torch.from_numpy(canvas).float().permute(2,0,1)/255
            # save_image([canvas_tensor, image_tensor_org, target_image, target_image2], nrow=1, fp=file_path)
            # save_image([canvas_tensor, image_tensor_org, target_image], nrow=1, fp=file_path)
            if imge_changed_flag:
                save_image([canvas_tensor, target_image], nrow=1, fp=file_path)
            else:
                save_image([canvas_tensor, image_tensor], nrow=1, fp=file_path)
            # print()

        

        ################  finish  then output  #############
        if imge_changed_flag:
            image_aug = target_image
            image_aug_np_rgb_255 = (image_aug*255).permute(1,2,0).numpy().astype(np.uint8)
            return image_aug_np_rgb_255
        else:
            # no change in image
            return image

     

def get_control_point(
    corners, 
    bboxes_center, 
    lidar2img,
    transform_matrix,
    img_height,
    img_width,
    imge_changed_flag):

    continue_flag = False

    target_points = None
    source_points = None
    right_line_index = None
    left_line_index = None
    mid_line_index = None

    for ixxxx in [0]:
        clockwise_order_corners = corners[[0,1,2,3,6,7,4,5]]

        clockwise_order_corners_xy = clockwise_order_corners[[0,2,4,6],:2]
        clockwise_order_corners_theta = safe_arctan_0to2pi(clockwise_order_corners_xy)
        for adjust_step in range(10):
            if clockwise_order_corners_theta.max() - clockwise_order_corners_theta.min() > np.pi:
                clockwise_order_corners_theta[clockwise_order_corners_theta.argmin()] += 2 * np.pi
            else:
                break
        if adjust_step > 3:
            print('error bbox may cover xyz-axis origin')
            continue_flag = True
            continue
        
        max_theta_idx = clockwise_order_corners_theta.argmax()
        min_theta_idx = clockwise_order_corners_theta.argmin()
        idx_minus = max_theta_idx - min_theta_idx
        if idx_minus.abs()==1 or idx_minus.abs()==3:
            only_one_face_saw = True
        else:
            only_one_face_saw = False

        if only_one_face_saw == False:
            clockwise_order_corners_d = clockwise_order_corners_xy.pow(2).sum(1)
            clockwise_order_corners_dmax_index = clockwise_order_corners_d.argmax()
            clockwise_order_corners_xy_near_mask = torch.ones_like(clockwise_order_corners_d).bool().fill_(True)
            clockwise_order_corners_xy_near_mask[clockwise_order_corners_dmax_index] = False
            clockwise_order_corners_theta_near = clockwise_order_corners_theta[clockwise_order_corners_xy_near_mask]
            clockwise_order_corners_theta_near_sort = clockwise_order_corners_theta_near.sort()[0]
            theta_part_length = clockwise_order_corners_theta_near_sort[1:] - clockwise_order_corners_theta_near_sort[:-1]
            
            if theta_part_length.min() / theta_part_length.sum() < 0.2:
                only_one_face_saw = True
        # compute control points
        if only_one_face_saw:
            corners_distence_xy_pow2 = corners[:,:2].pow(2).sum(1)
            nearest_4_corners_indices = (-corners_distence_xy_pow2).topk(k=4)[1]
            nearest_4_corners_xyz = corners[nearest_4_corners_indices]
            nearest_4_corners_uvd = points_cam2img(nearest_4_corners_xyz, proj_mat=lidar2img, with_depth=True)
            nearest_4_corners_uv = nearest_4_corners_uvd[:,:2]
            source_points = nearest_4_corners_uv # 4*2
            u_valid = (nearest_4_corners_uvd[:,0]>=0) * (nearest_4_corners_uvd[:,0]<img_width)
            v_valid = (nearest_4_corners_uvd[:,1]>=0) * (nearest_4_corners_uvd[:,1]<img_height)
            d_valid = (nearest_4_corners_uvd[:,2]>=0)
            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()

            if not uvd_valid: # 
                continue_flag = True
                continue
            else:
                imge_changed_flag = True

            nearest_4_corners_xyz_temp = nearest_4_corners_xyz - bboxes_center
            nearest_4_corners_xyz_temp = nearest_4_corners_xyz_temp.mm(transform_matrix)
            target_4_corners_xyz = nearest_4_corners_xyz_temp + bboxes_center

            target_4_corners_uv = points_cam2img(target_4_corners_xyz, proj_mat=lidar2img, with_depth=False)
            target_points = target_4_corners_uv

        else:
            
            corners_distence_xy_pow2 = corners[:,:2].pow(2).sum(1)
            nearest_6_corners_indices = (-corners_distence_xy_pow2).topk(k=6)[1]
            nearest_6_corners_xyz = corners[nearest_6_corners_indices]
            nearest_6_corners_uvd = points_cam2img(nearest_6_corners_xyz, proj_mat=lidar2img, with_depth=True)
            nearest_6_corners_uv = nearest_6_corners_uvd[:,:2]
            source_points = nearest_6_corners_uv # 6*2

            u_valid = (nearest_6_corners_uvd[:,0]>=0) * (nearest_6_corners_uvd[:,0]<img_width)
            v_valid = (nearest_6_corners_uvd[:,1]>=0) * (nearest_6_corners_uvd[:,1]<img_height)
            d_valid = (nearest_6_corners_uvd[:,2]>=0)

            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()
            if not uvd_valid:
                continue_flag = True
                continue
            else:
                imge_changed_flag = True

            nearest_6_corners_xyz_temp = nearest_6_corners_xyz - bboxes_center
            nearest_6_corners_xyz_temp = nearest_6_corners_xyz_temp.mm(transform_matrix)
            target_6_corners_xyz = nearest_6_corners_xyz_temp + bboxes_center

            target_6_corners_uv = points_cam2img(target_6_corners_xyz, proj_mat=lidar2img, with_depth=False)
            target_points = target_6_corners_uv

        # on box points
        # sort points
        if len(source_points) == 4:
            points_u_sort_index = torch.sort(source_points[:,0])[1]

            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)

            right_line_index =  points_u_sort_index[2:4]
            right_line_index = check_order_v(right_line_index, source_points)
        else:
            points_u_sort_index = torch.sort(source_points[:,0])[1]

            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)
            
            mid_line_index = points_u_sort_index[2:4]
            mid_line_index = check_order_v(mid_line_index, source_points)

            right_line_index =  points_u_sort_index[4:6]
            right_line_index = check_order_v(right_line_index, source_points)



    return continue_flag, imge_changed_flag, target_points, source_points, right_line_index, left_line_index, mid_line_index



def get_control_point_mono(
    corners, 
    bboxes_center, 
    cam2img,
    transform_matrix,
    img_height,
    img_width,
    imge_changed_flag):

    continue_flag = False

    target_points = None
    source_points = None
    right_line_index = None
    left_line_index = None
    mid_line_index = None

    for ixxxx in [0]:
        # clockwise_order_corners = corners[[0,1,2,3,6,7,4,5]]
        clockwise_order_corners = corners[[0,3,1,2,5,6,4,7]]
        clockwise_order_corners_xz = clockwise_order_corners[[0,2,4,6]][:,[0,2]]
        clockwise_order_corners_theta = safe_arctan_0to2pi(clockwise_order_corners_xz)
        for adjust_step in range(10):
            if clockwise_order_corners_theta.max() - clockwise_order_corners_theta.min() > np.pi:
                clockwise_order_corners_theta[clockwise_order_corners_theta.argmin()] += 2 * np.pi
            else:
                break
        if adjust_step > 3:
            print('error bbox may cover xyz-axis origin')
            continue_flag = True
            continue
        max_theta_idx = clockwise_order_corners_theta.argmax()
        min_theta_idx = clockwise_order_corners_theta.argmin()
        idx_minus = max_theta_idx - min_theta_idx
        if idx_minus.abs()==1 or idx_minus.abs()==3:
            only_one_face_saw = True
        else:
            only_one_face_saw = False
        if only_one_face_saw == False:
            clockwise_order_corners_d = clockwise_order_corners_xz.pow(2).sum(1)
            clockwise_order_corners_dmax_index = clockwise_order_corners_d.argmax()
            clockwise_order_corners_xz_near_mask = torch.ones_like(clockwise_order_corners_d).bool().fill_(True)
            clockwise_order_corners_xz_near_mask[clockwise_order_corners_dmax_index] = False
            clockwise_order_corners_theta_near = clockwise_order_corners_theta[clockwise_order_corners_xz_near_mask]
            clockwise_order_corners_theta_near_sort = clockwise_order_corners_theta_near.sort()[0]
            theta_part_length = clockwise_order_corners_theta_near_sort[1:] - clockwise_order_corners_theta_near_sort[:-1]
            
            if theta_part_length.min() / theta_part_length.sum() < 0.2:
                only_one_face_saw = True
        if only_one_face_saw == False:
            corners_distence_xz_pow2 = corners[:,[0,2]].pow(2).sum(1)
            nearest_6_corners_indices = (-corners_distence_xz_pow2).topk(k=6)[1]
            nearest_6_corners_mask = torch.zeros_like(corners_distence_xz_pow2).bool().fill_(False)
            nearest_6_corners_mask[nearest_6_corners_indices] = True

            corners_transformed = corners - bboxes_center
            corners_transformed = corners_transformed.mm(transform_matrix)
            corners_transformed = corners_transformed + bboxes_center

            corners_transformed_distence_xz_pow2 = corners_transformed[:,[0,2]].pow(2).sum(1)
            nearest_6_corners_transformed_indices = (-corners_transformed_distence_xz_pow2).topk(k=6)[1]
            nearest_6_corners_transformed_mask = torch.zeros_like(corners_transformed_distence_xz_pow2).bool().fill_(False)
            nearest_6_corners_transformed_mask[nearest_6_corners_transformed_indices] = True

            if (nearest_6_corners_transformed_mask == nearest_6_corners_mask).all():
                pass
            else:
                only_one_face_saw = True

        # compute control points
        if only_one_face_saw:
            corners_distence_xz_pow2 = corners[:,[0,2]].pow(2).sum(1)
            nearest_4_corners_indices = (-corners_distence_xz_pow2).topk(k=4)[1]
            nearest_4_corners_xyz = corners[nearest_4_corners_indices]
            nearest_4_corners_uvd = points_cam2img(nearest_4_corners_xyz, proj_mat=cam2img, with_depth=True)
            nearest_4_corners_uv = nearest_4_corners_uvd[:,:2]
            source_points = nearest_4_corners_uv # 4*2
            u_valid = (nearest_4_corners_uvd[:,0]>=0) * (nearest_4_corners_uvd[:,0]<img_width)
            v_valid = (nearest_4_corners_uvd[:,1]>=0) * (nearest_4_corners_uvd[:,1]<img_height)
            d_valid = (nearest_4_corners_uvd[:,2]>=0)
            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()

            if not uvd_valid: # 
                continue_flag = True
                continue
            else:
                imge_changed_flag = True

            nearest_4_corners_xyz_temp = nearest_4_corners_xyz - bboxes_center
            nearest_4_corners_xyz_temp = nearest_4_corners_xyz_temp.mm(transform_matrix)
            target_4_corners_xyz = nearest_4_corners_xyz_temp + bboxes_center

            target_4_corners_uv = points_cam2img(target_4_corners_xyz, proj_mat=cam2img, with_depth=False)
            target_points = target_4_corners_uv

        else:
            
            corners_distence_xz_pow2 = corners[:,[0,2]].pow(2).sum(1)
            nearest_6_corners_indices = (-corners_distence_xz_pow2).topk(k=6)[1]
            nearest_6_corners_xyz = corners[nearest_6_corners_indices]
            nearest_6_corners_uvd = points_cam2img(nearest_6_corners_xyz, proj_mat=cam2img, with_depth=True)
            nearest_6_corners_uv = nearest_6_corners_uvd[:,:2]
            source_points = nearest_6_corners_uv # 6*2

            u_valid = (nearest_6_corners_uvd[:,0]>=0) * (nearest_6_corners_uvd[:,0]<img_width)
            v_valid = (nearest_6_corners_uvd[:,1]>=0) * (nearest_6_corners_uvd[:,1]<img_height)
            d_valid = (nearest_6_corners_uvd[:,2]>=0)

            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()
            if not uvd_valid:
                continue_flag = True
                continue
            else:
                imge_changed_flag = True

            nearest_6_corners_xyz_temp = nearest_6_corners_xyz - bboxes_center
            nearest_6_corners_xyz_temp = nearest_6_corners_xyz_temp.mm(transform_matrix)
            target_6_corners_xyz = nearest_6_corners_xyz_temp + bboxes_center

            target_6_corners_uv = points_cam2img(target_6_corners_xyz, proj_mat=cam2img, with_depth=False)
            target_points = target_6_corners_uv

        # on box points

        # sort points
        if len(source_points) == 4:
            points_u_sort_index = torch.sort(source_points[:,0])[1]

            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)

            right_line_index =  points_u_sort_index[2:4]
            right_line_index = check_order_v(right_line_index, source_points)


        else:
            points_u_sort_index = torch.sort(source_points[:,0])[1]

            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)
            
            mid_line_index = points_u_sort_index[2:4]
            mid_line_index = check_order_v(mid_line_index, source_points)

            right_line_index =  points_u_sort_index[4:6]
            right_line_index = check_order_v(right_line_index, source_points)


    return continue_flag, imge_changed_flag, target_points, source_points, right_line_index, left_line_index, mid_line_index






def obj_img_transform(
        image_tensor,
        imge_changed_flag, target_points, source_points, 
        right_line_index, left_line_index, mid_line_index,
        watch_img, canvas,
        smaller_flag
    ):
    # whether to add more control points
    # 2 : no add
    # 3,4... : add
    total_num_inline = 2
    if len(source_points) == 4:
        source_points_onbox_leftright = get_grid_points(
            torch.cat([
                source_points[left_line_index],
                source_points[right_line_index],
            ]),
            total_num_inline
        )
        target_points_onbox_leftright = get_grid_points(
            torch.cat([
                target_points[left_line_index],
                target_points[right_line_index],
            ]),
            total_num_inline
        )

    else:
        source_points_onbox_leftmid = get_grid_points(
            torch.cat([
                source_points[left_line_index],
                source_points[mid_line_index],
            ]),
            total_num_inline
        )
        source_points_onbox_midright = get_grid_points(
            torch.cat([
                source_points[mid_line_index],
                source_points[right_line_index],
            ]),
            total_num_inline
        )
        target_points_onbox_leftmid = get_grid_points(
            torch.cat([
                target_points[left_line_index],
                target_points[mid_line_index],
            ]),
            total_num_inline
        )
        target_points_onbox_midright = get_grid_points(
            torch.cat([
                target_points[mid_line_index],
                target_points[right_line_index],
            ]),
            total_num_inline
        )

    # do seperate transform!
    # left--mid   mid--right

    if len(source_points) == 4:
        source_points_leftright = torch.cat([source_points_onbox_leftright, ])
        target_points_leftright = torch.cat([target_points_onbox_leftright, ])

    else:
        source_points_leftmid = torch.cat([source_points_onbox_leftmid, ])
        source_points_midright = torch.cat([source_points_onbox_midright, ])
        target_points_leftmid = torch.cat([target_points_onbox_leftmid, ])
        target_points_midright = torch.cat([target_points_onbox_midright, ])


    if watch_img:

        for i in range(0,len(source_points),2):
            u1 = int(source_points[i,0].long())
            v1 = int(source_points[i,1].long())
            u2 = int(source_points[i+1,0].long())
            v2 = int(source_points[i+1,1].long())
            cv2.line(canvas,(u1,v1),(u2,v2),color=(0,255,0),thickness=2)

        for i in range(0,len(target_points),2):
            u1 = int(target_points[i,0].long())
            v1 = int(target_points[i,1].long())
            u2 = int(target_points[i+1,0].long())
            v2 = int(target_points[i+1,1].long())
            cv2.line(canvas,(u1,v1),(u2,v2),color=(255,0,0),thickness=2)
        
        if len(source_points) == 4:

            for i in range(len(source_points_leftright)):
                u1 = int(source_points_leftright[i,0].long())
                v1 = int(source_points_leftright[i,1].long())
                u2 = int(target_points_leftright[i,0].long())
                v2 = int(target_points_leftright[i,1].long())
                cv2.circle(canvas, (u1,v1), 2, color=(100,255,0), thickness=-1)
                cv2.circle(canvas, (u2,v2), 2, color=(0,50,200), thickness=-1)

        else:
            for i in range(len(source_points_leftmid)):
                u1 = int(source_points_leftmid[i,0].long())
                v1 = int(source_points_leftmid[i,1].long())
                u2 = int(target_points_leftmid[i,0].long())
                v2 = int(target_points_leftmid[i,1].long())
                cv2.circle(canvas, (u1,v1), 2, color=(100,255,0), thickness=-1)
                cv2.circle(canvas, (u2,v2), 2, color=(0,50,200), thickness=-1)

    if len(source_points) == 4:
        u1_leftright, v1_leftright = torch.cat([
            target_points_onbox_leftright, source_points_onbox_leftright
        ]).min(0)[0].long()


        u2_leftright, v2_leftright = torch.cat([
            target_points_onbox_leftright, source_points_onbox_leftright
        ]).max(0)[0].long()



        # !=-------------------  input ------------------------------------
        image_tensor = patch_transform(
            u1=u1_leftright,
            u2=u2_leftright,
            v1=v1_leftright,
            v2=v2_leftright,
            target_points=target_points_leftright,
            source_points=source_points_leftright,
            image_tensor=image_tensor,
            smaller_flag=smaller_flag
        )


    else:
        u1_leftmid, v1_leftmid = torch.cat([
            target_points_onbox_leftmid, source_points_onbox_leftmid
        ]).min(0)[0].long()

        u2_leftmid, _ = torch.cat([
            target_points_onbox_leftmid, 
        ]).max(0)[0].long()

        _, v2_leftmid = torch.cat([
            target_points_onbox_leftmid, source_points_onbox_leftmid
        ]).max(0)[0].long()



        # !=-------------------  input ------------------------------------
        image_tensor = patch_transform(
            u1=u1_leftmid,
            u2=u2_leftmid,
            v1=v1_leftmid,
            v2=v2_leftmid,
            target_points=target_points_leftmid,
            source_points=source_points_leftmid,
            image_tensor=image_tensor,
            smaller_flag=smaller_flag
        )

        _, v1_midright = torch.cat([
            target_points_onbox_midright, source_points_onbox_midright
        ]).min(0)[0].long()

        u1_midright = u2_leftmid

        u2_midright, v2_midright = torch.cat([
            target_points_onbox_midright, source_points_onbox_midright
        ]).max(0)[0].long()

        image_tensor = patch_transform(
            u1=u1_midright,   # u1_midright == u2_midright
            u2=u2_midright,
            v1=v1_midright,
            v2=v2_midright,
            target_points=target_points_midright,
            source_points=source_points_midright,
            image_tensor=image_tensor,
            smaller_flag=smaller_flag
        )

    # replace
    target_image = image_tensor
    # image_tensor = target_image
    return target_image, canvas, imge_changed_flag

        
        


def patch_transform(
        u1, u2,
        v1, v2,
        target_points,
        source_points,
        image_tensor,
        smaller_flag
    ):

    try:
        if smaller_flag:
            v1_larger = v1 - (v2 - v1) * 0.5
            v2_larger = v2 + (v2 - v1) * 0.5

            u1_larger = u1 - (u2 - u1) * 0.5
            u2_larger = u2 + (u2 - u1) * 0.5

            v1_larger = v1_larger.int()
            v2_larger = v2_larger.int()
            u1_larger = u1_larger.int()
            u2_larger = u2_larger.int()
            target_height_loacl = v2_larger - v1_larger
            target_width_loacl = u2_larger - u1_larger
            target_points_local = target_points.clone()
            target_points_local[:,0] -= u1_larger
            target_points_local[:,1] -= v1_larger
            norm_target_points_local = target_points_local.clone()
            norm_target_points_local[:,0] = target_points_local[:,0]*2/target_width_loacl - 1
            norm_target_points_local[:,1] = target_points_local[:,1]*2/target_height_loacl - 1

            source_points_local = source_points.clone()
            source_points_local[:,0] -= u1_larger
            source_points_local[:,1] -= v1_larger
            norm_source_points_local = source_points_local.clone()
            norm_source_points_local[:,0] = source_points_local[:,0]*2/target_width_loacl - 1
            norm_source_points_local[:,1] = source_points_local[:,1]*2/target_height_loacl - 1
            


            tps = TPSGridGen(  
                            target_height_loacl, 
                            target_width_loacl, 
                            target_control_points=norm_target_points_local
                            )
            source_coordinate = tps(norm_source_points_local.unsqueeze(0))
            grid_local = source_coordinate.view(1, target_height_loacl, target_width_loacl, 2)
            source_image_local = safe_img_part_get_apply(
                image=image_tensor,
                x1=u1_larger,
                x2=u2_larger,
                y1=v1_larger,
                y2=v2_larger
            )

            target_image_local = F.grid_sample(source_image_local[None], grid_local, padding_mode="border", align_corners=False).squeeze()
            local_center_u1 = int((u2 - u1) * 0.5)
            local_center_v1 = int((v2 - v1) * 0.5)
            target_image_local_center = target_image_local[
                :,
                local_center_v1:local_center_v1 + (v2 - v1),
                local_center_u1:local_center_u1 + (u2 - u1)
            ]
            target_image = safe_img_patch_apply(
                image=image_tensor, 
                patch=target_image_local_center, 
                x1=u1, 
                y1=v1
                )
            image_tensor = target_image

        else:
            target_height_loacl = v2 - v1
            target_width_loacl = u2 - u1
            target_points_local = target_points.clone()
            target_points_local[:,0] -= u1
            target_points_local[:,1] -= v1
            norm_target_points_local = target_points_local.clone()
            norm_target_points_local[:,0] = target_points_local[:,0]*2/target_width_loacl - 1
            norm_target_points_local[:,1] = target_points_local[:,1]*2/target_height_loacl - 1

            source_points_local = source_points.clone()
            source_points_local[:,0] -= u1
            source_points_local[:,1] -= v1
            norm_source_points_local = source_points_local.clone()
            norm_source_points_local[:,0] = source_points_local[:,0]*2/target_width_loacl - 1
            norm_source_points_local[:,1] = source_points_local[:,1]*2/target_height_loacl - 1
  
            tps = TPSGridGen(  
                            target_height_loacl, 
                            target_width_loacl, 
                            target_control_points=norm_target_points_local
                            )
            source_coordinate = tps(norm_source_points_local.unsqueeze(0))
            grid_local = source_coordinate.view(1, target_height_loacl, target_width_loacl, 2)
            # source_image_local = image_tensor[:, v1:v2, u1:u2]
            source_image_local = safe_img_part_get_apply(
                image=image_tensor,
                x1=u1,
                x2=u2,
                y1=v1,
                y2=v2
            )

            target_image_local = F.grid_sample(source_image_local[None], grid_local, padding_mode="border", align_corners=False).squeeze()
            local_center_u1 = int((u2 - u1) * 0.5)
            local_center_v1 = int((v2 - v1) * 0.5)
            target_image_local_center = target_image_local[
                :,
            ]
            target_image = safe_img_patch_apply(
                image=image_tensor, 
                patch=target_image_local_center, 
                x1=u1, 
                y1=v1
                )
            image_tensor = target_image

    except:
        image_tensor = image_tensor

    return image_tensor





def get_4corner(points):
    uv_min = points.min(0)[0]
    uv_max = points.max(0)[0]
    u1,v1 = uv_min
    u2,v2 = uv_max
    return torch.Tensor([
        [u1,v1],
        [u1,v2],
        [u2,v1],
        [u2,v2],
    ])







class ImageBBoxMotionBlurFrontBack(): 
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity-1]

    def __call__(self, image, lidar2img, bboxes_centers, bboxes_corners, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        img_height=image.shape[0]
        img_width=image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0],canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]   # 8*3
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=lidar2img, with_depth=True)
            corners_uv = corners_uvd[:,:2]
            corners_depth = corners_uvd[:,2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)

            mask = mask + mask_temp
        mask_bool_float = (mask>0).astype(np.float32)[:,:,None]
        image_aug_layer = self.zoom_blur(image_rgb_255, corrpution)
        images_aug = image_aug_layer * mask_bool_float + (1-mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)

        if watch_img:
            mask_bool_float_tensor = torch.from_numpy(mask_bool_float).float().permute(2,0,1).repeat(3,1,1)
            images_aug_tensor = torch.from_numpy(images_aug).float().permute(2,0,1)/255
            save_image([mask_bool_float_tensor, images_aug_tensor], nrow=1, fp=file_path)
            # print()
        return image_aug_np_rgb_255

    def zoom_blur(self, x, corrpution):
        if corrpution <= 0.02:
            c = np.arange(1, 1+corrpution, 0.005)
        else:
            c = np.linspace(1, 1+corrpution, 4)
        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)

        set_exception = False
        for zoom_factor in c:
            if len(x.shape) < 3 or x.shape[2] < 3:
                x_channels = np.array([x, x, x]).transpose((1, 2, 0))
                zoom_layer = self.clipped_zoom(x_channels, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
            else:
                zoom_layer = self.clipped_zoom(x, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

            try:
                out += zoom_layer
            except ValueError:
                set_exception = True
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

        if set_exception:
            print('ValueError for zoom blur, Exception handling')
        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def clipped_zoom(self, img, zoom_factor):
        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2
        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                    (zoom_factor, zoom_factor, 1), order=1)
        return img



class ImageBBoxMotionBlurFrontBackMono(): 
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity-1]

    def __call__(self, image, cam2img, bboxes_centers, bboxes_corners, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        img_height=image.shape[0]
        img_width=image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0],canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]   # 8*3
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=cam2img, with_depth=True)
            corners_uv = corners_uvd[:,:2]
            corners_depth = corners_uvd[:,2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)

            mask = mask + mask_temp
        mask_bool_float = (mask>0).astype(np.float32)[:,:,None]
        image_aug_layer = self.zoom_blur(image_rgb_255, corrpution)
        images_aug = image_aug_layer * mask_bool_float + (1-mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)

        if watch_img:
            mask_bool_float_tensor = torch.from_numpy(mask_bool_float).float().permute(2,0,1).repeat(3,1,1)
            images_aug_tensor = torch.from_numpy(images_aug).float().permute(2,0,1)/255
            save_image([mask_bool_float_tensor, images_aug_tensor], nrow=1, fp=file_path)
            # print()
        return image_aug_np_rgb_255

    def zoom_blur(self, x, corrpution):
        c = np.arange(1, 1+corrpution, 0.005)
        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)

        set_exception = False
        for zoom_factor in c:
            if len(x.shape) < 3 or x.shape[2] < 3:
                x_channels = np.array([x, x, x]).transpose((1, 2, 0))
                zoom_layer = self.clipped_zoom(x_channels, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
            else:
                zoom_layer = self.clipped_zoom(x, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

            try:
                out += zoom_layer
            except ValueError:
                set_exception = True
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

        if set_exception:
            print('ValueError for zoom blur, Exception handling')
        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def clipped_zoom(self, img, zoom_factor):
        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2
        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                    (zoom_factor, zoom_factor, 1), order=1)
        return img


class ImageBBoxMotionBlurLeftRight(): 
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity-1]

    def __call__(self, image, lidar2img, bboxes_centers, bboxes_corners, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        img_height=image.shape[0]
        img_width=image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image


        # corruption
        img_width = image.shape[1]
        kernel_size = corrpution * img_width * 0.5
        kernel_size = int(kernel_size)
        self.iaa_seq = iaa.Sequential([
            iaa.MotionBlur(k=kernel_size, angle=90),
        ])
        # bboxes_corners 3*8*3   tensor 
        # lidar2img 4*4   np.array
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0],canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]   # 8*3
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=lidar2img, with_depth=True)
            corners_uv = corners_uvd[:,:2]
            corners_depth = corners_uvd[:,2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)

            mask = mask + mask_temp
        mask_bool_float = (mask>0).astype(np.float32)[:,:,None]
        images_rgb_255 = image_rgb_255[None]
        image_aug_layer = self.iaa_seq(images=images_rgb_255)[0]
        images_aug = image_aug_layer * mask_bool_float + (1-mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)

        if watch_img:
            mask_bool_float_tensor = torch.from_numpy(mask_bool_float).float().permute(2,0,1).repeat(3,1,1)
            images_aug_tensor = torch.from_numpy(images_aug).float().permute(2,0,1)/255
            save_image([mask_bool_float_tensor, images_aug_tensor], nrow=1, fp=file_path)
        return image_aug_np_rgb_255




class ImageBBoxMotionBlurLeftRightMono(): 
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity-1]

    def __call__(self, image, cam2img, bboxes_centers, bboxes_corners, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        img_height=image.shape[0]
        img_width=image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image


        # corruption
        img_width = image.shape[1]
        kernel_size = corrpution * img_width * 0.5
        kernel_size = int(kernel_size)
        self.iaa_seq = iaa.Sequential([
            iaa.MotionBlur(k=kernel_size, angle=90),
        ])
        # bboxes_corners 3*8*3   tensor 
        # cam2img 4*4   np.array
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0],canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]   # 8*3
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=cam2img, with_depth=True)
            corners_uv = corners_uvd[:,:2]
            corners_depth = corners_uvd[:,2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)

            mask = mask + mask_temp
        mask_bool_float = (mask>0).astype(np.float32)[:,:,None]
        images_rgb_255 = image_rgb_255[None]
        image_aug_layer = self.iaa_seq(images=images_rgb_255)[0]
        images_aug = image_aug_layer * mask_bool_float + (1-mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)

        if watch_img:
            mask_bool_float_tensor = torch.from_numpy(mask_bool_float).float().permute(2,0,1).repeat(3,1,1)
            images_aug_tensor = torch.from_numpy(images_aug).float().permute(2,0,1)/255
            save_image([mask_bool_float_tensor, images_aug_tensor], nrow=1, fp=file_path)
        return image_aug_np_rgb_255






class ImageMotionBlurFrontBack():
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity-1]



    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        corrpution = self.corrpution

        image_rgb_255 = image
        images_aug = self.zoom_blur(image_rgb_255, corrpution)
        image_aug_rgb_255 = images_aug
        if watch_img:
            save_image(torch.from_numpy(image_aug_rgb_255).permute(2,0,1).float() /255., file_path)
        return image_aug_rgb_255

    def zoom_blur(self, x, corrpution):
        c = np.arange(1, 1+corrpution, 0.005)
        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)

        set_exception = False
        for zoom_factor in c:
            if len(x.shape) < 3 or x.shape[2] < 3:
                x_channels = np.array([x, x, x]).transpose((1, 2, 0))
                zoom_layer = self.clipped_zoom(x_channels, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
            else:
                zoom_layer = self.clipped_zoom(x, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]

            try:
                out += zoom_layer
            except ValueError:
                set_exception = True
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer

        if set_exception:
            print('ValueError for zoom blur, Exception handling')
        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def clipped_zoom(self, img, zoom_factor):
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2
        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                    (zoom_factor, zoom_factor, 1), order=1)
        return img


class ImageMotionBlurLeftRight():
    def __init__(self, severity, corrput_list=[0.02*i for i in range(1,6)]) -> None:
        self.severity = severity
        self.corrput_list = corrput_list

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        img_width = image.shape[1]
        kernel_size=self.corrput_list[self.severity-1] * img_width * 0.5
        kernel_size = int(kernel_size)
        self.iaa_seq = iaa.Sequential([
            iaa.MotionBlur(k=kernel_size, angle=90),
        ])
        image_rgb_255 = image
        # iaa requires rgb_255_uint8 img
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        if watch_img:
            save_image(torch.from_numpy(image_aug_rgb_255).permute(2,0,1).float() /255., file_path)
        return image_aug_rgb_255



class ImageAddGaussianNoise():
    def __init__(self, severity, seed) -> None:
        self.iaa_seq = iaa.Sequential([
            iaa.imgcorruptlike.GaussianNoise(severity=severity, seed=seed),
        ])

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        if watch_img:
            save_image(torch.from_numpy(image_aug_rgb_255).permute(2,0,1).float() /255., file_path)
        return image_aug_rgb_255



class ImageAddImpulseNoise():
    def __init__(self, severity, seed) -> None:
        self.iaa_seq = iaa.Sequential([
            iaa.imgcorruptlike.ImpulseNoise(severity=severity, seed=seed),
        ])

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_rgb_255 = image
        # iaa requires rgb_255_uint8 img
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        if watch_img:
            save_image(torch.from_numpy(image_aug_rgb_255).permute(2,0,1).float() /255., file_path)
        return image_aug_rgb_255


class ImageAddUniformNoise():
    def __init__(self, severity) -> None:
        print(self.__class__.__name__, ': please set numpy seed !')
        self.severity = severity

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_rgb_255 = image
        severity = self.severity
        # iaa requires rgb_255_uint8 img
        image_aug_rgb_255 = self.uniform_noise(image_rgb_255, severity)
        if watch_img:
            save_image(torch.from_numpy(image_aug_rgb_255).permute(2,0,1).float() /255., file_path)
        return image_aug_rgb_255

    def uniform_noise(self, x, severity):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.
        return (np.clip(x + np.random.uniform(low=-c, high=c, size=x.shape), 0, 1) * 255).astype(np.uint8)
        



class ImageAddSnow():
    def __init__(self, severity, seed) -> None:
        self.iaa_seq = iaa.Sequential([
            iaa.imgcorruptlike.Snow(severity=severity, seed=seed),
        ])

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        
        # add snow
        # iaa requires rgb_255_uint8 img
        images = image[None]
        images_aug = self.iaa_seq(images=images)
        image_aug = images_aug[0]

        # be gray-like
        gray_ratio = 0.3
        image_aug = gray_ratio * np.ones_like(image_aug)*128 \
            + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)


        # lower the brightness
        image_rgb_255 = image_aug
        img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
        img_hsv[:,:,2] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 256 * 0.7
        img_hsv[:,:,2] = np.clip(img_hsv[:,:,2], 0,255)
        img_hsv = img_hsv.astype(np.uint8)
        image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        image_aug = image_rgb_255


        if watch_img:
            save_image(torch.from_numpy(image_aug).permute(2,0,1).float() /255., file_path)
        return image_aug



class ImageAddFog():
    def __init__(self, severity, seed) -> None:
        self.iaa_seq = iaa.Sequential([
            iaa.imgcorruptlike.Fog(severity=severity, seed=seed),
        ])
        self.gray_ratio = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ][severity-1]

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        image_aug = image_aug_rgb_255

        # be gray-like
        gray_ratio = self.gray_ratio
        image_aug = gray_ratio * np.ones_like(image_aug)*128 \
            + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)

        if watch_img:
            save_image(torch.from_numpy(image_aug).permute(2,0,1).float() /255., file_path)
        return image_aug



class ImageAddRain():
    def __init__(self, severity, seed) -> None:
        density = [
            (0.01,0.06),
            (0.06,0.10),
            (0.10,0.15),
            (0.15,0.20),
            (0.20,0.25),
        ][severity-1]
        self.iaa_seq = iaa.Sequential([
            iaa.RainLayer(
                density=density,
                density_uniformity=(0.8, 1.0),  
                drop_size=(0.4, 0.6),  
                drop_size_uniformity=(0.2, 0.5),  
                angle=(-15,15),   
                speed=(0.04, 0.20),  
                blur_sigma_fraction=(0.0001,0.001),   
                blur_sigma_limits=(0.5, 3.75),   
                seed=seed
            )
        ])

    def __call__(self, image, watch_img=False, file_path='') -> np.array:
        """
            image should be numpy array : H * W * 3
            in uint8 (0~255) and RGB
        """
        # add rain
        # iaa requires rgb_255_uint8 img
        images = image[None]
        images_aug = self.iaa_seq(images=images)
        image_aug = images_aug[0]

        # be gray-like
        gray_ratio = 0.3
        image_aug = gray_ratio * np.ones_like(image_aug)*128 \
            + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)

        # lower the brightness
        image_rgb_255 = image_aug
        img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
        img_hsv[:,:,2] = img_hsv[:,:,2] / img_hsv[:,:,2].max() * 256 * 0.7
        img_hsv[:,:,2] = np.clip(img_hsv[:,:,2], 0,255)
        img_hsv = img_hsv.astype(np.uint8)
        image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        image_aug = image_rgb_255


        if watch_img:
            save_image(torch.from_numpy(image_aug).permute(2,0,1).float() /255., file_path)
        return image_aug


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def read_kitti_info(calib_path, extend_matrix):
    calib_info = {}
    info = {}
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                    ]).reshape([3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                    ]).reshape([3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                    ]).reshape([3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                    ]).reshape([3, 4])
    # P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
    #                 ]).reshape([3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
        # P4 = _extend_matrix(P4)
    R0_rect = np.array([
        float(info) for info in lines[5-1].split(' ')[1:10]
    ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect

    Tr_velo_to_cam = np.array([
        float(info) for info in lines[6-1].split(' ')[1:13]
    ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
    calib_info['P0'] = P0
    calib_info['P1'] = P1
    calib_info['P2'] = P2
    calib_info['P3'] = P3
    # calib_info['P4'] = P4
    calib_info['R0_rect'] = rect_4x4
    calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
    info['calib'] = calib_info
    return info


def safe_arctan_0to2pi(xy):
    """
        xy: (n, 2)
        return: (n) in [0,2pi)
    """
    x = xy[:,0]
    y = xy[:,1]
    safe_mask = (x!=0)*(y!=0)

    safe_x = x[safe_mask]
    safe_y = y[safe_mask]
    safe_quadrant_0_mask = (safe_x>0)*(safe_y>0)
    safe_quadrant_1_mask = (safe_x<0)*(safe_y>0)
    safe_quadrant_2_mask = (safe_x<0)*(safe_y<0)
    safe_quadrant_3_mask = (safe_x>0)*(safe_y<0)


    arctan_value = torch.zeros_like(x)
    

    safe_arctan_value = \
        safe_quadrant_0_mask * torch.atan(safe_y/safe_x) \
        + safe_quadrant_1_mask * (torch.atan(safe_y/safe_x) + np.pi)\
        + safe_quadrant_2_mask * (torch.atan(safe_y/safe_x) + np.pi)\
        + safe_quadrant_3_mask * (torch.atan(safe_y/safe_x) + 2*np.pi)

    arctan_value[safe_mask] = safe_arctan_value

    axis_0_mask = (x>0)*(y==0)
    axis_1_mask = (x==0)*(y>0)
    axis_2_mask = (x<0)*(y==0)
    axis_3_mask = (x==0)*(y<0)

    origin_mask = (x==0)*(y==0)

    arctan_value[origin_mask] = 0
    arctan_value[axis_0_mask] = 0
    arctan_value[axis_1_mask] = np.pi / 2
    arctan_value[axis_2_mask] = np.pi
    arctan_value[axis_3_mask] = np.pi /2 *3

    return arctan_value


def check_order_v(index, points):
    if points[index][0,1] > points[index][1,1]:
        index_new = index
    else:
        index_new = index[[1,0]]
    return index_new

def get_grid_points(points4, point_num=4):
    '''
        0----2
        |    |
        1----3
    '''
    new_points_list=[]
    total_line_seg_num = point_num - 1
    for i in range(0, point_num):
        alpha = i/total_line_seg_num
        line_point_a = points4[0]*alpha + points4[1]*(1-alpha)
        if len(points4) == 2:
            print()
        line_point_b = points4[2]*alpha + points4[3]*(1-alpha)
        for j in range(0,point_num):
            beta = j/total_line_seg_num
            new_point = line_point_a*beta + line_point_b*(1-beta)
            new_points_list.append(new_point)
    new_points_tensor = torch.stack(new_points_list)
    return new_points_tensor



def safe_img_patch_apply(image, patch, x1, y1):
    assert len(image.shape) == len(patch.shape)
    assert image.shape[0] == patch.shape[0]

    try:
        w_patch = patch.shape[-1]
        h_patch = patch.shape[-2]
        x2 = x1 + w_patch
        y2 = y1 + h_patch
        w_img = image.shape[-1]
        h_img = image.shape[-2]

        x1_use = x1
        x2_use = x2
        y1_use = y1
        y2_use = y2
        patch_use = patch

        if x1 < 0:
            x1_use = 0
            patch_use = patch_use[:, :, -x1:]
        elif x1 > w_img-1:
            return image

        if x2 > w_img:
            x2_use = w_img
            patch_use = patch_use[:, :, :-(x2 - w_img)]
        elif x2 < 0:
            return image

        if y1 < 0:
            y1_use = 0
            patch_use = patch_use[:, -y1:, :]
        elif y1 > h_img - 1:
            return image

        if y2 > h_img:
            y2_use = h_img
            patch_use = patch_use[:, :-(y2 - h_img), :]
        elif y2 < 0:
            return image

        image[:, y1_use:y2_use, x1_use:x2_use] = patch_use
    except:
        return image
    return image




def safe_img_part_get_apply(image, x1, x2, y1, y2):
    # inverse imgpart get
    # regard image as a patch and use the other function
    x1_inverse = 0 - x1
    y1_inverse = 0 - y1
    canvas = image.new_zeros(image.shape[0], y2 - y1, x2 - x1)
    out = safe_img_patch_apply(
        image=canvas,
        patch=image,
        x1=x1_inverse,
        y1=y1_inverse
    )
    return out