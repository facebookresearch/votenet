# Deep Hough Voting for 3D Object Detection in Point Clouds
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="https://orlitany.github.io/" target="_blank">Or Litany</a>, <a href="http://kaiminghe.com/" target="_blank">Kaiming He</a> and <a href="https://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas Guibas</a> from <a href="https://research.fb.com/category/facebook-ai-research/" target="_blank">Facebook AI Research</a> and <a href="http://www.stanford.edu" target="_blank">Stanford University</a>.

![teaser](https://github.com/facebookresearch/votenet/blob/master/doc/teaser.jpg)

## Introduction
This repository is code release for our ICCV 2019 paper (arXiv report [here](https://arxiv.org/pdf/1904.09664.pdf)).

Current 3D object detection methods are heavily influenced by 2D detectors. In order to leverage architectures in 2D detectors, they often convert 3D point clouds to regular grids (i.e., to voxel grids or to bird’s eye view images), or rely on detection in 2D images to propose 3D boxes. Few works have attempted to directly detect objects in point clouds. In this work, we return to first principles to construct a 3D detection pipeline for point cloud data and as generic as possible. However, due to the sparse nature of the data – samples from 2D manifolds in 3D space – we face a major challenge when directly predicting bounding box parameters from scene points: a 3D object centroid can be far from any surface point thus hard to regress accurately in one step. To address the challenge, we propose VoteNet, an end-to-end 3D object detection network based on a synergy of deep point set networks and Hough voting. Our model achieves state-of-the-art 3D detection on two large datasets of real 3D scans, ScanNet and SUN RGB-D with a simple design, compact model size and high efficiency. Remarkably, VoteNet outperforms previous methods by using purely geometric information without relying on color images.

In this repository, we provide VoteNet model implementation (with Pytorch) as well as data preparation, training and evaluation scripts on SUN RGB-D and ScanNet.

## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{qi2019deep,
        author = {Qi, Charles R and Litany, Or and He, Kaiming and Guibas, Leonidas J},
        title = {Deep Hough Voting for 3D Object Detection in Point Clouds},
        booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
        year = {2019}
    }

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: there is some incompatibility with newer version of Pytorch (e.g. v1.3), which is to be fixed.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'

## Run demo

You can download pre-trained models and sample point clouds [HERE](https://drive.google.com/file/d/1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l/view?usp=sharing).
Unzip the file under the project root path (`/path/to/project/demo_files`) and then run:

    python demo.py

The demo uses a pre-trained model (on SUN RGB-D) to detect objects in a point cloud from an indoor room of a table and a few chairs (from SUN RGB-D val set). You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the dumped file under `demo_files/sunrgbd_results` to see the 3D detection output. Specifically, open `***_pc.ply` and `***_pred_confident_nms_bbox.ply` to see the input point cloud and predicted 3D bounding boxes.

You can also run the following command to use another pretrained model on a ScanNet:

    python demo.py --dataset scannet --num_point 40000

Detection results will be dumped to `demo_files/scannet_results`.

## Training and evaluating

### Data preparation

For SUN RGB-D, follow the [README](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) under the `sunrgbd` folder.

For ScanNet, follow the [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md) under the `scannet` folder.

### Train and test on SUN RGB-D

To train a new VoteNet model on SUN RGB-D data (depth images):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options (e.g. you can also set `--model boxnet` to train with the baseline BoxNet model).
While training you can check the `log_sunrgbd/log_train.txt` file on its progress, or use the TensorBoard to see loss curves.

To test the trained model with its checkpoint:

    python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_sunrgbd` folder (or any other folder you specify). You can run `python eval.py -h` to see the full options for evaluation. After the evaluation, you can use MeshLab to visualize the predicted votes and 3D bounding boxes (select wireframe mode to view the boxes).
Final evaluation results will be printed on screen and also written in the `log_eval.txt` file under the dump directory. In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes. A properly trained VoteNet should have around 57 mAP@0.25 and 32 mAP@0.5.

### Train and test on ScanNet

To train a VoteNet model on Scannet data (fused scan):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_scannet --num_point 40000

To test the trained model with its checkpoint:

    python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on axis aligned boxes. A properly trained VoteNet should have around 58 mAP@0.25 and 35 mAP@0.5.

### Train on your own data

[For Pro Users] If you have your own dataset with point clouds and annotated 3D bounding boxes, you can create a new dataset class and train VoteNet on your own data. To ease the proces, some tips are provided in this [doc](https://github.com/facebookresearch/votenet/blob/master/doc/tips.md).

## Acknowledgements
We want to thank Erik Wijmans for his PointNet++ implementation in Pytorch ([original codebase](https://github.com/erikwijmans/Pointnet2_PyTorch)).

## License
votenet is relased under the MIT License. See the [LICENSE file](https://arxiv.org/pdf/1904.09664.pdf) for more details.

## Change log
10/20/2019: Fixed a bug of the 3D interpolation customized ops (corrected gradient computation). Re-training the model after the fix slightly improves mAP (less than 1 point).
