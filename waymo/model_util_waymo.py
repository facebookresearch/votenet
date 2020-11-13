import numpy as np


class WaymoDatasetConfig(object):

    def __init__(self):
        # TODO not clear if this is off by 1 we have 3 classes + no class
        self.num_class = 3
        self.num_heading_bin = 1
        self.num_size_cluster = 3

        self.type2class = {
            'SIGN': 3,
            'CAR': 1,
            'PERSON': 2
        }

        self.class2type = {self.type2class[t] for t in self.type2class}
        self.waymoDataIds = np.array([0, 1, 2, 3])
        # TODO
        # self.mean_size_arr = {
        #     1: np.array([1.94191027, 4.41998532, 1.97638222]),
        #     2: np.array([0.8433135,  0.90025193, 1.71406664]),
        #     3: np.array([0.63715241, 0.45786768, 0.76657946]),
        # }

        self.mean_size_arr = np.array(
            [
                [1.94191027, 4.41998532, 1.97638222],
                [0.8433135, 0.90025193, 1.71406664],
                [0.63715241, 0.45786768, 0.76657946],
            ]
        )

        self.type_mean_size = {}
        # TODO
        # for i in range(self.num_size_cluster):
        #     self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert (False)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb


def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)