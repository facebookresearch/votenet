# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Testing customized ops. '''

import torch
from torch.autograd import gradcheck
import numpy as np

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet2_utils

def test_interpolation_grad():
    batch_size = 1
    feat_dim = 2
    m = 4
    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True).float().cuda()
    
    def interpolate_func(inputs):
        idx = torch.from_numpy(np.array([[[0,1,2],[1,2,3]]])).int().cuda()
        weight = torch.from_numpy(np.array([[[1,1,1],[2,2,2]]])).float().cuda()
        interpolated_feats = pointnet2_utils.three_interpolate(inputs, idx, weight)
        return interpolated_feats
    
    assert (gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1))

if __name__=='__main__':
    test_interpolation_grad()
