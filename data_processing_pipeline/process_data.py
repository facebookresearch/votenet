import os
import sys
import tensorflow as tf
import math
import numpy as np
import itertools

from itertools import islice

from uuid import uuid4


from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


from os import listdir
from os.path import isfile, join, split


#### TQDM JOBLIB CODE
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# Import code for multiprocessing
from joblib import Parallel, delayed

DEFAULT_FILE_PREFIX = "/home/bcj/stanford/cs230/project/waymo_data/"

BASE_OUTPUT_DIR = "./output_data/final_dataset_w_frusts/"

DEFAULT_OUTPUT_FILE_TEMPLATE = "{scan_name}_{data_type}.npy"

FRUSTUM_OUTPUT_FILE_TEMPLATE = "{scan_name}_{data_type}_FRUSTUM.npy"


from collections import defaultdict
from waymo_open_dataset.utils.box_utils import is_within_box_3d
TYPE_2_SIZE_DICT = defaultdict(lambda: [])

def prep_data(frame, num_points = 40000):
    mesh_vertices, instance_labels, semantic_labels, instance_bboxes = None, None, None, None
    
    # Transform frame to point cloud
    (range_images, camera_projections,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose) 
    
    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    
    # sub sample points
    if num_points < 0 :
        mesh_vertices = points_all
    else:
        sub_samp_points = points_all[np.random.choice(np.arange(points_all.shape[0]), num_points),:]
        mesh_vertices = sub_samp_points
    
    
    num_detected_objects = len(frame.laser_labels)
    
    instance_labels = np.zeros((len(mesh_vertices),))
    semantic_labels = np.zeros((len(mesh_vertices),))
    instance_bboxes = np.zeros((num_detected_objects, 8))


    instance_id_map = {}
    
    # loop through each ground truth object
    for instance_id, detected_object in enumerate(frame.laser_labels):
        if detected_object.type not in instance_id_map:
            instance_id_map[detected_object.type] = (instance_id, detected_object)
    
        box = detected_object.box
        
        TYPE_2_SIZE_DICT[detected_object.type].append(
            np.array([box.width,
            box.length,
            box.height,])
        )
        points_in_bbox = is_within_box_3d(point=tf.convert_to_tensor(mesh_vertices, dtype=tf.float32), box=tf.convert_to_tensor(np.array([
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            box.heading,
        ]).reshape((1,7)), dtype=tf.float32)).numpy().ravel()
    
        
        instance_labels[points_in_bbox] = instance_id + 1 # The reason we do this is so that we don't treat not being part of an isntance as being part of instance zero
        semantic_labels[points_in_bbox] = detected_object.type
         
        instance_bboxes[instance_id,:] = np.array([
            box.center_x,
            box.center_y,
            box.center_z,
            box.width,
            box.length,
            box.height,
            box.heading,
            detected_object.type - 1, # Map to 0 - num_classes -1 vs. 1 - num_classes
        ]).reshape((1,8))
        
        
        # Do a check if we have all points
        if num_points < 0:
            print(np.sum(points_in_bbox > 0 ), detected_object.num_lidar_points_in_box)

    # Return full scene results
    yield ('FULL_SCENE', (mesh_vertices, instance_labels, semantic_labels, instance_bboxes))

    # Loop through and generate data for each frustum ... TODO do we need to do any padding !
    for instance_id, detected_object in instance_id_map.values():
        box = detected_object.box
        center_vec = np.array([
            box.center_x,
            box.center_y,
            box.center_z,
        ])
        
        normalized_cone_dic_vec = center_vec / np.linalg.norm(center_vec)
        
        # Go thorugh all the frusutum bull shit
        box = detected_object.box
        near_dist = np.linalg.norm(np.array([
            box.center_x,
            box.center_y,
            box.center_z,
        ]))
        

        fov_angle = 2* np.arctan(box.height / (2* near_dist))
        aspect_ratio = box.width / box.height
        
        
        base_radius = max(box.height, box.width) * 1.5 # The base radius of the cone is max of length / width of the far frustum with a 10% fudge factor

        # this is very inefficent ... lets vectorize this code
        mesh_vertices_mask = np.zeros((mesh_vertices.shape[0])).astype(np.int32)
        for i in range(mesh_vertices.shape[0]):
            p = mesh_vertices[i,:]
            # TODO need to do this for all objects in the scene
            cone_dist = np.dot(p, normalized_cone_dic_vec)
            cone_radius = (cone_dist / near_dist) * base_radius
            
            orth_distance = np.linalg.norm(p - cone_dist * normalized_cone_dic_vec)
            mesh_vertices_mask[i] =  orth_distance < cone_radius

        yield ('FRUSTUM', (mesh_vertices[mesh_vertices_mask == 1,:], instance_labels[mesh_vertices_mask == 1], semantic_labels[mesh_vertices_mask == 1], instance_bboxes[instance_id,:]))

            

            


            
            
    
    # return mesh_vertices, instance_labels, semantic_labels, instance_bboxes
   


def _process_single(data):
    """

    Process a single tf record entry, writing the result to disk

    """
    
    # Read data from tfrecord
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    scan_name = frame.context.name.replace('_','X') + 'FRAMENUM{}'.format(str(uuid4()))   
    # process frame into data format we want
    # mesh_vertices, instance_labels, semantic_labels, instance_bboxes = prep_data(frame, 150000)

    for result in prep_data(frame, 150000):
        tag, data = result
        mesh_vertices, instance_labels, semantic_labels, instance_bboxes = data 
        scan_name = frame.context.name.replace('_','X') + 'FRAMENUM{}'.format(str(uuid4()))   
        if tag != 'FRUSTUM':
            FILENAME_TEMPLATE = BASE_OUTPUT_DIR + DEFAULT_OUTPUT_FILE_TEMPLATE
            ## Write mesh verticies
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="vert"), 'wb+') as f:
                np.save(f, mesh_vertices)
                
            ## Write instance labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="ins_label"), 'wb+') as f:
                np.save(f, instance_labels)
                
            ## Write semantic labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="sem_label"), 'wb+') as f:
                np.save(f, semantic_labels)
            
            ## Write instance_bboxes labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="bbox"), 'wb+') as f:
                np.save(f, instance_bboxes)
        else:
            FILENAME_TEMPLATE = BASE_OUTPUT_DIR + FRUSTUM_OUTPUT_FILE_TEMPLATE
            ## Write mesh verticies
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="vert"), 'wb+') as f:
                np.save(f, mesh_vertices)
                
            ## Write instance labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="ins_label"), 'wb+') as f:
                np.save(f, instance_labels)
                
            ## Write semantic labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="sem_label"), 'wb+') as f:
                np.save(f, semantic_labels)
            
            ## Write instance_bboxes labels
            with open(FILENAME_TEMPLATE.format(scan_name=scan_name, data_type="bbox"), 'wb+') as f:
                np.save(f, instance_bboxes)


def process_data(fileprefix=DEFAULT_FILE_PREFIX):
    """
    Processes all tf records into npy files accepted by votenet in Parallel

    :param fileprefix str

    :returns: None
    """

    # TODO wow this is uggo code
    FILE_PREFIX = fileprefix

    MAX_SAMP=1500

    # Get data from file
    tf_record_file_names = [join(FILE_PREFIX, f) for f in listdir(FILE_PREFIX) if isfile(join(FILE_PREFIX, f)) and 'tfrecord' in f]
    assert len(tf_record_file_names) > 0

    dataset_it = iter(tf.data.TFRecordDataset(tf_record_file_names, compression_type='').take(MAX_SAMP))

    # Run the computation !
    with tqdm_joblib(tqdm(desc="My calculation", total=MAX_SAMP)) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(_process_single)(data) for data in dataset_it
        )









if __name__ == "__main__":
    # Optionally accept cli file prefix
    if len(sys.argv) > 1:
        process_data(fileprefix=sys.arvg[1])
    else:
        process_data()




