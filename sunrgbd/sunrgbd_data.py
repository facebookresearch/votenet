# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''

import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import sunrgbd_utils

DEFAULT_TYPE_WHITELIST = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training', use_v1=False):
        self.root_dir = root_dir
        self.split = split
        assert(self.split=='training') 
        self.split_dir = os.path.join(root_dir)

        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        if use_v1:
            self.label_dir = os.path.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = os.path.join(self.split_dir, 'label')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return sunrgbd_utils.load_image(img_filename)

    def get_depth(self, idx): 
        depth_filename = os.path.join(self.depth_dir, '%06d.mat'%(idx))
        return sunrgbd_utils.load_depth_points_mat(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return sunrgbd_utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return sunrgbd_utils.read_sunrgbd_label(label_filename)

def data_viz(data_dir, dump_dir=os.path.join(BASE_DIR, 'data_viz_dump')):  
    ''' Examine and visualize SUN RGB-D data. '''
    sunrgbd = sunrgbd_object(data_dir)
    idxs = np.array(range(1,len(sunrgbd)+1))
    np.random.seed(0)
    np.random.shuffle(idxs)
    for idx in range(len(sunrgbd)):
        data_idx = idxs[idx]
        print('-'*10, 'data index: ', data_idx)
        pc = sunrgbd.get_depth(data_idx)
        print('Point cloud shape:', pc.shape)
        
        # Project points to image
        calib = sunrgbd.get_calibration(data_idx)
        uv,d = calib.project_upright_depth_to_image(pc[:,0:3])
        print('Point UV:', uv)
        print('Point depth:', d)
        
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
        
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i in range(uv.shape[0]):
            depth = d[i]
            color = cmap[int(120.0/depth),:]
            cv2.circle(img, (int(np.round(uv[i,0])), int(np.round(uv[i,1]))), 2,
                color=tuple(color), thickness=-1)
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        Image.fromarray(img).save(os.path.join(dump_dir,'img_depth.jpg'))
        
        # Load box labels
        objects = sunrgbd.get_label_objects(data_idx)
        print('Objects:', objects)
        
        # Draw 2D boxes on image
        img = sunrgbd.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i,obj in enumerate(objects):
            cv2.rectangle(img, (int(obj.xmin),int(obj.ymin)),
                (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
            cv2.putText(img, '%d %s'%(i,obj.classname), (max(int(obj.xmin),15),
                max(int(obj.ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,0,0), 2)
        Image.fromarray(img).save(os.path.join(dump_dir, 'img_box2d.jpg'))
       
        # Dump OBJ files for the colored point cloud 
        for num_point in [10000,20000,40000,80000]:
            sampled_pcrgb = pc_util.random_sampling(pc, num_point)
            pc_util.write_ply_rgb(sampled_pcrgb[:,0:3],
                (sampled_pcrgb[:,3:]*256).astype(np.int8),
                os.path.join(dump_dir, 'pcrgb_%dk.obj'%(num_point//1000)))
        # Dump OBJ files for 3D bounding boxes
        # l,w,h correspond to dx,dy,dz
        # heading angle is from +X rotating towards -Y
        # (+X is degree, -Y is 90 degrees)
        oriented_boxes = []
        for obj in objects:
            obb = np.zeros((7))
            obb[0:3] = obj.centroid
            # Some conversion to map with default setting of w,l,h
            # and angle in box dumping
            obb[3:6] = np.array([obj.l,obj.w,obj.h])*2
            obb[6] = -1 * obj.heading_angle
            print('Object cls, heading, l, w, h:',\
                 obj.classname, obj.heading_angle, obj.l, obj.w, obj.h)
            oriented_boxes.append(obb)
        if len(oriented_boxes)>0:
            oriented_boxes = np.vstack(tuple(oriented_boxes))
            pc_util.write_oriented_bbox(oriented_boxes,
                os.path.join(dump_dir, 'obbs.ply'))
        else:
            print('-'*30)
            continue

        # Draw 3D boxes on depth points
        box3d = []
        ori3d = []
        for obj in objects:
            corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d(obj, calib)
            ori_3d_image, ori_3d = sunrgbd_utils.compute_orientation_3d(obj, calib)
            print('Corners 3D: ', corners_3d)
            box3d.append(corners_3d)
            ori3d.append(ori_3d)
        pc_box3d = np.concatenate(box3d, 0)
        pc_ori3d = np.concatenate(ori3d, 0)
        print(pc_box3d.shape)
        print(pc_ori3d.shape)
        pc_util.write_ply(pc_box3d, os.path.join(dump_dir, 'box3d_corners.ply'))
        pc_util.write_ply(pc_ori3d, os.path.join(dump_dir, 'box3d_ori.ply'))
        print('-'*30)
        print('Point clouds and bounding boxes saved to PLY files under %s'%(dump_dir))
        print('Type anything to continue to the next sample...')
        input()

def extract_sunrgbd_data(idx_filename, split, output_folder, num_point=20000,
    type_whitelist=DEFAULT_TYPE_WHITELIST,
    save_votes=False, use_v1=False, skip_empty_scene=True):
    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        idx_filename: a TXT file where each line is an int number (index)
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.
        use_v1: use the SUN RGB-D V1 data
        skip_empty_scene: if True, skip scenes that contain no object (no objet in whitelist)

    Dumps:
        <id>_pc.npz of (N,6) where N is for number of subsampled points and 6 is
            for XYZ and RGB (in 0~1) in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 8 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    dataset = sunrgbd_object('./sunrgbd_trainval', split, use_v1=use_v1)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        objects = dataset.get_label_objects(data_idx)

        # Skip scenes with 0 object
        if skip_empty_scene and (len(objects)==0 or \
            len([obj for obj in objects if obj.classname in type_whitelist])==0):
                continue

        object_list = []
        for obj in objects:
            if obj.classname not in type_whitelist: continue
            obb = np.zeros((8))
            obb[0:3] = obj.centroid
            # Note that compared with that in data_viz, we do not time 2 to l,w.h
            # neither do we flip the heading angle
            obb[3:6] = np.array([obj.l,obj.w,obj.h])
            obb[6] = obj.heading_angle
            obb[7] = sunrgbd_utils.type2class[obj.classname]
            object_list.append(obb)
        if len(object_list)==0:
            obbs = np.zeros((0,8))
        else:
            obbs = np.vstack(object_list) # (K,8)

        pc_upright_depth = dataset.get_depth(data_idx)
        pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

        np.savez_compressed(os.path.join(output_folder,'%06d_pc.npz'%(data_idx)),
            pc=pc_upright_depth_subsampled)
        np.save(os.path.join(output_folder, '%06d_bbox.npy'%(data_idx)), obbs)
       
        if save_votes:
            N = pc_upright_depth_subsampled.shape[0]
            point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
            point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
            indices = np.arange(N)
            for obj in objects:
                if obj.classname not in type_whitelist: continue
                try:
                    # Find all points in this object's OBB
                    box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(obj.centroid,
                        np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
                    pc_in_box3d,inds = sunrgbd_utils.extract_pc_in_box3d(\
                        pc_upright_depth_subsampled, box3d_pts_3d)
                    # Assign first dimension to indicate it is in an object box
                    point_votes[inds,0] = 1
                    # Add the votes (all 0 if the point is not in any object's OBB)
                    votes = np.expand_dims(obj.centroid,0) - pc_in_box3d[:,0:3]
                    sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                    for i in range(len(sparse_inds)):
                        j = sparse_inds[i]
                        point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                        # Populate votes with the fisrt vote
                        if point_vote_idx[j] == 0:
                            point_votes[j,4:7] = votes[i,:]
                            point_votes[j,7:10] = votes[i,:]
                    point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
                except:
                    print('ERROR ----',  data_idx, obj.classname)
            np.savez_compressed(os.path.join(output_folder, '%06d_votes.npz'%(data_idx)),
                point_votes = point_votes)

    
def get_box3d_dim_statistics(idx_filename,
    type_whitelist=DEFAULT_TYPE_WHITELIST,
    save_path=None):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    dataset = sunrgbd_object('./sunrgbd_trainval')
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue
            heading_angle = -1 * np.arctan2(obj.orientation[1], obj.orientation[0])
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.classname) 
            ry_list.append(heading_angle)

    import cPickle as pickle
    if save_path is not None:
        with open(save_path,'wb') as fp:
            pickle.dump(type_list, fp)
            pickle.dump(dimension_list, fp)
            pickle.dump(ry_list, fp)

    # Get average box size for different catgories
    box3d_pts = np.vstack(dimension_list)
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i]==class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])
        median_box3d = np.median(box3d_list,0)
        print("\'%s\': np.array([%f,%f,%f])," % \
            (class_type, median_box3d[0]*2, median_box3d[1]*2, median_box3d[2]*2))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', help='Run data visualization.')
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument('--gen_v1_data', action='store_true', help='Generate V1 dataset.')
    parser.add_argument('--gen_v2_data', action='store_true', help='Generate V2 dataset.')
    args = parser.parse_args()

    if args.viz:
        data_viz(os.path.join(BASE_DIR, 'sunrgbd_trainval'))
        exit()

    if args.compute_median_size:
        get_box3d_dim_statistics(os.path.join(BASE_DIR, 'sunrgbd_trainval/train_data_idx.txt'))
        exit()

    if args.gen_v1_data:
        extract_sunrgbd_data(os.path.join(BASE_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v1_train'),
            save_votes=True, num_point=50000, use_v1=True, skip_empty_scene=False)
        extract_sunrgbd_data(os.path.join(BASE_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v1_val'),
            save_votes=True, num_point=50000, use_v1=True, skip_empty_scene=False)
    
    if args.gen_v2_data:
        extract_sunrgbd_data(os.path.join(BASE_DIR, 'sunrgbd_trainval/train_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v2_train'),
            save_votes=True, num_point=50000, use_v1=False, skip_empty_scene=False)
        extract_sunrgbd_data(os.path.join(BASE_DIR, 'sunrgbd_trainval/val_data_idx.txt'),
            split = 'training',
            output_folder = os.path.join(BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v2_val'),
            save_votes=True, num_point=50000, use_v1=False, skip_empty_scene=False)
