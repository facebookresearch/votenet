import os
import sys
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
#from model_util_waymo import rotate_aligned_boxes
import sunrgbd_utils
from model_util_waymo import WaymoDatasetConfig

DC = WaymoDatasetConfig()
MAX_NUM_OBJ = 256
MEAN_COLOR_RGB = np.array(
    [109.8, 97.2, 83.8])  # TODO probably wrong ... tbh it does not matter ... tbh it does not matter

# docker run  --gpus all -it --ipc=host -v /home/ubuntu/data:/data waymo_votenet /bin/bash
class WaymoDetectionDataset(Dataset):

    def __init__(self, split_set='train', num_points=40000,
                 use_color=False, use_height=False, augment=False, data_path=None):

        # TODO if you have loading errors probably look here first!
        if not data_path:
            self.data_path = '/data'  # os.path.join('/work/data', 'waymo_train_dataset')
        else:
            self.data_path = data_path

        # TODO implement logic for split sets other than train
        self.scan_names = list(set([file.split('_')[0] for file in os.listdir(self.data_path) if 'npy' in file]))

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy').astype(np.int32)
        bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

            # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                       self.num_points, return_choices=True)
        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]
        #
        # pcl_color = pcl_color[choices]

        # target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        # target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            pass
            # if np.random.random() > 0.5:
            #     # Flipping along the YZ plane
            #     point_cloud[:, 0] = -1 * point_cloud[:, 0]
            #     target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
            #
            # if np.random.random() > 0.5:
            #     # Flipping along the XZ plane
            #     point_cloud[:, 1] = -1 * point_cloud[:, 1]
            #     target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
            #
            #     # Rotation along up-axis/Z-axis
            # rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            # rot_mat = pc_util.rotz(rot_angle)
            # point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            # target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)


        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1



        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in set(DC.type2class.values()):
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = semantic_labels[i]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            box3d_size = bbox[3:6]
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class + 1])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        return ret_dict

if __name__ == '__main__':
    dset = WaymoDetectionDataset(data_path=sys.argv[1])
    for i in range(1):
        example = dset.__getitem__(i)
        print(example)