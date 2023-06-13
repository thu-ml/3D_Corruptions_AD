import numpy as np
import math
import os
from pyquaternion import Quaternion
from functools import reduce
from torch.autograd import Variable

np.random.seed(1024)  # set the same seed
num_step = 100
kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def lidar_split(lidar, pts_pose):
    pose_matrix = np.squeeze(pts_pose)
    if not pose_matrix.shape[0] == 2:
        # we firstly transform the pc from kitti coordinate to nuscene coordinate
        start_pc = np.squeeze(lidar).T
        # <--be careful here! we need to convert to nuscene format
        start_pc = np.dot(
            kitti_to_nu_lidar.rotation_matrix, start_pc[:3, :])

        # change to polar coordinate
        polar_points = np.arctan2(
            start_pc[1, :], start_pc[0, :]) * 180 / np.pi + 180  # in degrees (0, 360]

        polar_points_min = np.floor(np.min(polar_points)-0.1)
        polar_points_max = np.ceil(np.max(polar_points))

        start_pose_rec_translation = [
            pose_matrix[0, 0], pose_matrix[0, 1], pose_matrix[0, 2]]
        start_pose_rec_rotation = [
            pose_matrix[0, 3], pose_matrix[0, 4], pose_matrix[0, 5], pose_matrix[0, 6]]

        start_cs_rec_translation = [
            pose_matrix[1, 0], pose_matrix[1, 1], pose_matrix[1, 2]]
        start_cs_rec_rotation = [
            pose_matrix[1, 3], pose_matrix[1, 4], pose_matrix[1, 5], pose_matrix[1, 6]]

        end_pose_rec_translation = [
            pose_matrix[2, 0], pose_matrix[2, 1], pose_matrix[2, 2]]
        end_pose_rec_rotation = [
            pose_matrix[2, 3], pose_matrix[2, 4], pose_matrix[2, 5], pose_matrix[2, 6]]

        # enable motion distortion
        # Init
        sensor_from_vehicle = transform_matrix(
            start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=True)
        vehicle_from_global = transform_matrix(
            start_pose_rec_translation, Quaternion(start_pose_rec_rotation), inverse=True)

        global_from_car = transform_matrix(
            start_pose_rec_translation, Quaternion(start_pose_rec_rotation), inverse=False)
        car_from_current = transform_matrix(
            start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=False)

        # find the next sample data
        translation_step = (np.array(
            end_pose_rec_translation) - np.array(start_pose_rec_translation))/num_step

        p_start = start_pose_rec_rotation
        q_end = end_pose_rec_rotation

        # trans_matrix_gps_list = list()
        pc_timestap_list = list()

        for t in range(num_step):
            t_current = start_pose_rec_translation + t * translation_step
            q_current = []

            cosa = p_start[0]*q_end[0] + p_start[1]*q_end[1] + \
                p_start[2]*q_end[2] + p_start[3]*q_end[3]

            # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
            # the shorter path. Fix by reversing one quaternion.
            if cosa < 0.0:
                q_end[0] = -q_end[0]
                q_end[1] = -q_end[1]
                q_end[2] = -q_end[2]
                q_end[3] = -q_end[3]
                cosa = -cosa

            # If the inputs are too close for comfort, linearly interpolate
            if cosa > 0.9995:
                k0 = 1.0 - t/num_step
                k1 = t/num_step
            else:
                sina = np.sqrt(1.0 - cosa*cosa)
                a = math.atan2(sina, cosa)
                k0 = math.sin((1.0 - t/num_step)*a) / sina
                k1 = math.sin(t*a/num_step) / sina

            q_current.append(p_start[0]*k0 + q_end[0]*k1)
            q_current.append(p_start[1]*k0 + q_end[1]*k1)
            q_current.append(p_start[2]*k0 + q_end[2]*k1)
            q_current.append(p_start[3]*k0 + q_end[3]*k1)

            ref_from_car = transform_matrix(
                start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=True)
            car_from_global = transform_matrix(
                t_current, Quaternion(q_current), inverse=True)

            # select the points in a small scan area
            small_delta = (polar_points_max-polar_points_min)/num_step

            scan_start = polar_points > small_delta*t + polar_points_min
            scan_end = polar_points <= small_delta*(t+1) + polar_points_min
            scan_area = np.logical_and(scan_start, scan_end)
            current_pc = start_pc[:, scan_area]

            # transform point cloud at start timestep into the interpolatation step t
            trans_matrix = reduce(
                np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc = trans_matrix.dot(
                np.vstack((current_pc, np.ones(current_pc.shape[1]))))
            pc_timestap_list.append(current_pc)

            '''
            Now calculate GPS compensation transformation
            '''
            vehicle_from_sensor = transform_matrix(
                start_cs_rec_translation, Quaternion(start_cs_rec_rotation), inverse=False)
            global_from_vehicle = transform_matrix(
                t_current, Quaternion(q_current), inverse=False)
            # can also calculate the inverse matrix of trans_matrix
            trans_matrix_gps = reduce(np.dot, [
                                        sensor_from_vehicle, vehicle_from_global, global_from_vehicle, vehicle_from_sensor])

            trans_matrix_gps = np.expand_dims(trans_matrix_gps, 0)

            if t == 0:
                trans_matrix_gps_tensor = trans_matrix_gps
            else:
                trans_matrix_gps_tensor = np.concatenate(
                    [trans_matrix_gps_tensor, trans_matrix_gps], 0)  # [1000, 4, 4]
    return trans_matrix_gps_tensor, pc_timestap_list



def reconstruct_pc(pc_timestap_list, trans_matrix_gps_tensor):
    import torch
    trans_matrix_gps_tensor = torch.Tensor(trans_matrix_gps_tensor).cuda()
    kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
    # aggregate the motion distortion points
    init_flag = False
    for timestap in range(100):
        pc_curr = pc_timestap_list[timestap]

        if not pc_curr.shape[1] == 0:
            pc_curr = torch.Tensor(pc_curr).cuda()
            tmp_pc = torch.mm(
                trans_matrix_gps_tensor[timestap, :], pc_curr)[:3, :]
            if init_flag == False:
                all_pc = tmp_pc
                init_flag = True
            else:
                all_pc = torch.cat((all_pc, tmp_pc), 1)  # 3 * 16384

    KITTI_to_NU_R = Variable(torch.Tensor(
        kitti_to_nu_lidar_inv.rotation_matrix))
    KITTI_to_NU_R = KITTI_to_NU_R.cuda()

    inputs = torch.mm(KITTI_to_NU_R, all_pc)
    pc_noise = torch.unsqueeze(inputs.transpose(0, 1), 0)
    return pc_noise   


def reconstruct_pc_cpu(pc_timestap_list, trans_matrix_gps_tensor):
    trans_matrix_gps_tensor = trans_matrix_gps_tensor
    kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
    # aggregate the motion distortion points
    init_flag = False
    for timestap in range(100):
        pc_curr = pc_timestap_list[timestap]

        if not pc_curr.shape[1] == 0:
            pc_curr = pc_curr
            tmp_pc = np.dot(
                trans_matrix_gps_tensor[timestap, :], pc_curr)[:3, :]
            if init_flag == False:
                all_pc = tmp_pc
                init_flag = True
            else:
                all_pc = np.concatenate((all_pc, tmp_pc), 1)  # 3 * 16384

    KITTI_to_NU_R = kitti_to_nu_lidar_inv.rotation_matrix
    # KITTI_to_NU_R = KITTI_to_NU_R

    inputs = np.dot(KITTI_to_NU_R, all_pc)
    pc_noise = inputs.transpose((1,0))
    return pc_noise   
