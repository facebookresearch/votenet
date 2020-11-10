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
        self.waymoDataIds = np.array([0,1,2,3])
        # TODO
        # self.mean_size_arr = {
        #     1: np.array([1.94191027, 4.41998532, 1.97638222]),
        #     2: np.array([0.8433135,  0.90025193, 1.71406664]),
        #     3: np.array([0.63715241, 0.45786768, 0.76657946]),
        # }

        self.mean_size_arr = np.array(
                [
                    [0,0,0],
                    [1.94191027, 4.41998532, 1.97638222],
                    [0.8433135,  0.90025193, 1.71406664],
                    [0.63715241, 0.45786768, 0.76657946],
                ]
        )

        self.type_mean_size = {}
        # TODO
        # for i in range(self.num_size_cluster):
        #     self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]
