import time
import pickle
from models import generator
from utils import DataLoader, load, save, psnr_error
from models import generator
from utils import DataLoader, load, save, psnr_error
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os 
import numpy as np
import cv2
import pandas as pd


def initialization():
    slim = tf.contrib.slim

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'#const.GPU

    dataset_name = 'cowdata'#const.DATASET
    test_folder = '../Data/'+dataset_name+'/testing/frames'#const.TEST_FOLDER
    DECIDABLE_IDX = 4
    num_his = 4#const.NUM_HIS
    height, width = 256, 256

    snapshot_dir = 'checkpoints/pretrains/'+dataset_name#const.SNAPSHOT_DIR
    # normalize scores in each sub video
    NORMALIZE = True
    # define dataset
    with tf.name_scope('dataset'):
        test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                                dtype=tf.float32)
        
        test_inputs = test_video_clips_tensor[..., 0:num_his*3]
        test_gt = test_video_clips_tensor[..., -3:]
        print('test inputs = {}'.format(test_inputs))
        print('test prediction gt = {}'.format(test_gt))

    # define testing generator function and
    # in testing, only generator networks, there is no discriminator networks and flownet.
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        print('testing = {}'.format(tf.get_variable_scope().name))
        test_outputs = generator(test_inputs, layers=4, output_channel=3)
        test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    
    returnValues = [test_video_clips_tensor, test_inputs, test_gt, test_outputs, test_psnr_error]
    return returnValues
        
