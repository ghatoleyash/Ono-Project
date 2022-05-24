"""
Loads the Generator model,
calculates the PSNR score in comparison
with the Ground Truth Frame and returns 
the decision for the input Frame
"""

import sys
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


dataset_name = 'cowdata'#const.DATASET
test_folder = '../Data/'+dataset_name+'/testing/frames'#const.TEST_FOLDER
DECIDABLE_IDX = 4
num_his = 4#const.NUM_HIS
height, width = 256, 256
NORMALIZE = True
THRESHOLD = 0.73 #Experimental


def inference(test_video_clips_tensor, test_inputs, test_gt, test_outputs, test_psnr_error):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # dataset
        data_loader = DataLoader(test_folder, height, width)
        data_loader.videos['01']['frame'] = sorted(data_loader.videos['01']['frame'], key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #print('Data Loader: ', data_loader.videos['01']['frame'])
        

        # initialize weights
        sess.run(tf.global_variables_initializer())
        #print('Init global successfully!')
        # tf saver
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
        restore_var = [v for v in tf.global_variables()]
        loader = tf.train.Saver(var_list=restore_var)
        ckpt = 'checkpoints/pretrains/'+dataset_name
        load(loader, sess, ckpt)


        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())  
        terminateFrame = num_his+1
        
        path = '../Codes/PSNRS.csv'
        isExist = os.path.exists(path)
        
        #Adding the frame result to the PSNRS.csv
        frames = os.listdir('../Data/'+dataset_name+'/testing/frames/01/')
        video_name = '01'
        start = num_his
        if isExist:
          start = len(frames) - 1
          psnrs = pd.read_csv(path)['PSNRS'].tolist()
          s = pd.read_csv(path)['SCORE'].tolist()
        else:
          psnrs = [0]*(terminateFrame)
          s = [1]*(terminateFrame)

        #Get the video frame for generating predictions
        video_clip = data_loader.get_video_clips(video_name, start - num_his, start+1)
        psnr = sess.run(test_psnr_error,feed_dict={test_video_clips_tensor:video_clip[np.newaxis, ...]})
        psnrs.append(psnr)

        if start==terminateFrame:
          psnrs[0:terminateFrame] = [psnr]*(terminateFrame)
          s[0:terminateFrame] = [1]*(terminateFrame)
        df = pd.DataFrame(psnrs, columns=['PSNRS'])
        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        
        # video normalization
        distance = psnrs
        if NORMALIZE:
          mini = min(distance)
          maxi = max(distance)
          distance[:] = [i-mini for i in distance]  # distances = (distance - min) / (max - min)
          distance[:] = [i/(maxi-mini) for i in distance]
          # distance = 1 - distance


        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
        
        print('video = {} / {}, i = {}, psnr = {:.6f}'.format(video_name, num_videos, start, psnr))
        print('SCORES: ', scores[-1])

        #Appending the result to the existing dataframe
        s.append(scores[-1])
        dfScore = pd.DataFrame(s, columns=['SCORE'])
        df = pd.concat([df, dfScore], axis=1)
        df.to_csv(path)

        #Tensorflow Session termination
        del sess

        if scores[-1]<THRESHOLD:
            return 'Anomaly'
        else:
            return 'Normal'
        
        
           
        
