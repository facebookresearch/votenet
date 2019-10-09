# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

import numpy as np
import pc_util

scene_name = 'scannet_train_detection_data/scene0002_00'
output_folder = 'data_viz_dump'

data = np.load(scene_name+'_vert.npy')
scene_points = data[:,0:3]
colors = data[:,3:]
instance_labels = np.load(scene_name+'_ins_label.npy')
semantic_labels = np.load(scene_name+'_sem_label.npy')
instance_bboxes = np.load(scene_name+'_bbox.npy')

print(np.unique(instance_labels))
print(np.unique(semantic_labels))
input()
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Write scene as OBJ file for visualization
pc_util.write_ply_rgb(scene_points, colors, os.path.join(output_folder, 'scene.obj'))
pc_util.write_ply_color(scene_points, instance_labels, os.path.join(output_folder, 'scene_instance.obj'))
pc_util.write_ply_color(scene_points, semantic_labels, os.path.join(output_folder, 'scene_semantic.obj'))

from model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
print(instance_bboxes.shape)
