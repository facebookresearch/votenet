from torch.utils.data import Dataset
import numpy as np
import os
import sys

import pc_util
# from model_util_waymo import WaymoDatasetConfig
# DC = WaymoDatasetConfig()
# TODO this is almost def wrong
# from model_util_scannet import ScannetDatasetConfig
# DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64 # TODO probably too low
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8]) # TODO probably wrong


class WaymoDetectionDataset(Dataset):

    def __init__(self, split_set='train', num_points=40000,
                 use_color=False, use_height=False, augment=False, data_path = None):

        # TODO if you have loading errors probably look here first!
        if not data_path:
            self.data_path = os.path.join('/work/data', 'waymo_train_dataset')
        else:
            self.data_path = data_path

        # TODO implement logic for split sets other than train
        self.scan_names = list(set([file.split('_')[0] for file in  os.listdir(self.data_path) if 'npy' in file]))

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        print(self.scan_names)
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')

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
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

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
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.nyu40id2class[x] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        return ret_dict

if __name__ == '__main__':
    dset = WaymoDetectionDataset(data_path=sys.argv[1])
    for i in range(4):
        example = dset.__getitem__(i)
        print(example)
