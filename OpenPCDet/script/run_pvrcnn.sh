#!/bin/bash

# add the severity wanted in severity list
severity=(1 2 3 4 5)

# add the corruptions wanted in map list
map=("lidar_crosstalk_noise" "density_dec_global" "gaussian_noise" "cutout_bbox" "density_dec_bbox" "gaussian_noise_bbox" "scale_bbox" "shear_bbox" "moving_noise_bbox")

for i in ${!map[@]}; do
    p_noise=${map[$i]}
    echo "run infer of $p_noise "
    for i in ${!severity[@]}; do
        echo "run infer of $p_noise and serverity of ${severity[$i]} "
        python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 8 --ckpt ../ckpt/pv_rcnn_8369.pth --corruptions=$p_noise --severity=${severity[$i]}
    done
done
