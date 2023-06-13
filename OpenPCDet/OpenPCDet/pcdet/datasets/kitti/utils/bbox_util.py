import numpy as np
import math
import os
import pyquaternion

'''
select bbox
'''


def check_point_in_box(pts, box):
    """
    	pts[x,y,z]
    	box[c_x,c_y,c_z,dx,dy,dz,heading]
    """

    shift_x = pts[0] - box[0]
    shift_y = pts[1] - box[1]
    shift_z = pts[2] - box[2]
    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx, dy, dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a
    if (abs(shift_z) > dz / 2.0 or abs(local_x) > dx / 2.0 or abs(local_y) > dy / 2.0):
        return False
    return True


def img2velodyne(calib_dir, img_id, p):
    """
    :param calib_dir
    :param img_id
    :param velo_box: (n,8,4)
    :return: (n,4)
    """
    calib_txt = os.path.join(calib_dir, img_id) + '.txt'
    calib_lines = [line.rstrip('\n') for line in open(calib_txt, 'r')]
    for calib_line in calib_lines:
        if 'P2' in calib_line:
            P2 = calib_line.split(' ')[1:]
            P2 = np.array(P2, dtype='float').reshape(3, 4)
        elif 'R0_rect' in calib_line:
            R0_rect = np.zeros((4, 4))
            R0 = calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3, :3] = R0
            R0_rect[-1, -1] = 1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam = calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3, :] = velo2cam
            velo_to_cam[-1, -1] = 1

    pts_rect_hom = p
    pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_rect, velo_to_cam).T))

    return pts_lidar


'''
Corruptions
'''


def density(pointcloud, severity):
    N, C = pointcloud.shape
    num = int(N * 0.1)
    c = [int(0.1 * N), int(0.2 * N), int(0.3 * N), int(0.4 * N), int(0.5 * N)][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud


def cutout(pointcloud, severity):
    N, C = pointcloud.shape
    # from 30 changed to 3000 to qualify kitti
    c = [(1, int(N * 0.3)), (1, int(N * 0.4)), (1, int(N * 0.5)), (1, int(N * 0.6)), (1, int(N * 0.7))][severity - 1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud


def gaussian(pointcloud, severity):
    N, C = pointcloud.shape  # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def uniform(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def impulse(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity - 1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud


'''
bbox_convert
'''


def to_Max2(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7
    Returns:
        points normalized to max-2 unit square box: N x 3+C
    """
    # shift
    points[:, :3] = points[:, :3] - gt_boxes_lidar[:3]
    # normalize to 2 units
    points[:, :3] = points[:, :3] / np.max(gt_boxes_lidar[3:6]) * 2
    # reversely rotate
    angle = -gt_boxes_lidar[6]
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points = np.hstack((points_rot, points[:, 3:].reshape(-1, 1)))

    return points


def to_Lidar(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7
    Returns:
        points denormalized to lidar coordinates
    """
    angle = gt_boxes_lidar[6]
    # along_z
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points = np.hstack((points_rot, points[:, 3:].reshape(-1, 1)))
    # denormalize to lidar
    points[:, :3] = points[:, :3] * np.max(gt_boxes_lidar[3:6]) / 2
    # shift
    points[:, :3] = points[:, :3] + gt_boxes_lidar[:3]

    return points


# normalize
def normalize_gt(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3
    Returns:
        limit points to gt: N x 3+C
    """
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio / np.max(gt_box_ratio)
        for i in range(3):
            indicator = np.max(np.abs(points[:, i])) / box_boundary_normalized[i]
            if indicator > 1:
                points[:, i] /= indicator
    return points


def shear(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity - 1]

    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    # shear
    b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    matrix = np.array([1, 0, b,
                       d, 1, e,
                       f, 0, 1]).reshape(3, 3)

    new_pc = np.matmul(pts_obj_max2[:, :3], matrix).astype('float32')

    pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:, 3].reshape(-1, 1)))
    pts_obj_max2_crp = normalize_gt(pts_obj_max2_crp, gt_boxes[3:6])
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor


def scale(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [0.04, 0.08, 0.12, 0.16, 0.20][severity - 1]
    xs_list, ys_list, zs_list = [], [], []

    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    ## scale on two randomly selected directions
    xs, ys, zs = 1.0, 1.0, 1.0
    r = np.random.randint(0, 3)
    t = np.random.choice([-1, 1])
    if r == 0:
        xs += c * t
    elif r == 1:
        ys += c * t
    else:
        zs += c * t
    matrix = np.array([[xs, 0, 0, 0], [0, ys, 0, 0], [0, 0, zs, 0], [0, 0, 0, 1]])
    pts_obj_max2_crp = np.matmul(pts_obj_max2, matrix)
    pts_obj_max2_crp[:, 2] += (zs - 1) * gt_boxes[5] / np.max(gt_boxes[3:6])
    xs_list.append(xs)
    ys_list.append(ys)
    zs_list.append(zs)
    # convert to Lidar
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor


def rotation(pointcloud, severity, gt_boxes):
    N, _ = pointcloud.shape
    c = [1, 3, 5, 7, 9][severity - 1]
    beta = np.random.uniform(c - 1, c + 1) * np.random.choice([-1, 1]) * np.pi / 180.
    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    ## rotation
    matrix_roration_z = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    pts_rotated = np.matmul(pts_obj_max2[:, :3], matrix_roration_z)
    pts_obj_max2_crp = np.hstack((pts_rotated, pts_obj_max2[:, 3].reshape(-1, 1)))
    # convert to lidar
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor


def moving_object(pointcloud, severity):
    # for kitti: the x is forward
    N, C = pointcloud.shape
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
    m1, m2 = float(c / 2), c
    x_min, x_max = min(pointcloud[:, 0]), max(pointcloud[:, 0])
    x_l = (x_max - x_min) / 3
    for i in range(len(pointcloud)):
        if pointcloud[i, 0] > x_min and pointcloud[i, 0] <= x_min + x_l:
            pointcloud[i, 0] += m1
        elif pointcloud[i, 0] <= x_max and pointcloud[i, 0] > x_min + x_l:
            pointcloud[i, 0] += m2
        else:
            continue
    return pointcloud


MAP = {
    'density_dec_bbox': density,
    'cutout_bbox': cutout,
    'gaussian_noise_bbox': gaussian,
    'uniform_noise_bbox': uniform,
    'impulse_noise_bbox': impulse,
    'scale_bbox': scale,
    'shear_bbox': shear,
    'rotation_bbox': rotation,
    'moving_noise_bbox': moving_object,
}


def pick_bbox(cor, slevel, data, pointcloud):

    # for openpcdet
    if len(data) == 1:

        idx = data
        xyz = pointcloud

        # for bbox
        f2 = open("./label_2/" + idx + '.txt')
        line2 = f2.readline()

        while line2:

            pcd_1 = []
            pcd_2 = []
            lt = line2.split()
            if lt[0] == 'DontCare' or lt[0] == 'Misc':
                line2 = f2.readline()
                continue

            H = float(lt[8])
            W = float(lt[9])
            L = float(lt[10])

            x = float(lt[11])
            y = float(lt[12])
            z = float(lt[13])
            angel = -(np.pi / 2 + float(lt[14]))

            p3 = (x, y, z, 1)
            p3 = img2velodyne('./calib', idx, p3)
            p3[2] = p3[2] + H / 2

            gt_boxes = []
            gt_boxes.append(p3[0])
            gt_boxes.append(p3[1])
            gt_boxes.append(p3[2])
            gt_boxes.append(L)
            gt_boxes.append(W)
            gt_boxes.append(H)
            gt_boxes.append(angel)

            for a in xyz:
                flag = check_point_in_box(a, gt_boxes)
                if flag == True:
                    pcd_2.append(a)
                else:
                    pcd_1.append(a)

            pcd_2 = np.array(pcd_2)
            if len(pcd_2) != 0:
                if 'bbox' in cor:
                    pcd_2 = MAP[cor](pcd_2, slevel, gt_boxes)
                else:
                    pcd_2 = MAP[cor](pcd_2, slevel)
                xyz = np.append(pcd_2, pcd_1, axis=0)

            line2 = f2.readline()
        f2.close()

    # for mmdetection3d
    else:
        xyz = pointcloud
        bboxes = data[0]
        for box in bboxes:
            pcd_1 = []
            pcd_2 = []
            x = float(box[0])
            y = float(box[1])
            z = float(box[2])
            x_size = float(box[3])
            y_size = float(box[4])
            z_size = float(box[5])
            angel = float(box[6])
            p3 = (x, y, z)
            gt_boxes = []
            gt_boxes.append(p3[0])
            gt_boxes.append(p3[1])
            gt_boxes.append(p3[2])
            gt_boxes.append(x_size)
            gt_boxes.append(y_size)
            gt_boxes.append(z_size)
            gt_boxes.append(angel)

            for a in xyz:
                flag = check_point_in_box(a, gt_boxes)
                if flag == True:
                    pcd_2.append(a)
                else:
                    pcd_1.append(a)
            pcd_2 = np.array(pcd_2)
            if len(pcd_2) != 0:
                if 'bbox' in cor:
                    pcd_2 = MAP[cor](pcd_2, slevel, gt_boxes)
                else:
                    pcd_2 = MAP[cor](pcd_2, slevel)
                xyz = np.append(pcd_2, pcd_1, axis=0)

    return xyz







