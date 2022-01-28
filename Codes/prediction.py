import os 
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import os
import time
import numpy as np
import pickle
import tensorflow as tf
import os
import time
import numpy as np
import pandas as pd
import pickle
from models import generator
from utils import DataLoader, load, save, psnr_error
from models import generator
from utils import DataLoader, load, save, psnr_error
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

def inference():
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

        ckpt = 'checkpoints/pretrains/ped1'
        load(loader, sess, ckpt)

        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())

        length = 5
        
        path = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Codes/PSNRS.csv'
        isExist = os.path.exists(path)
        
        frames = os.listdir('/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/01/')
        video_name = '01'
        start = num_his
        if isExist:
          start = len(frames) - 1
          psnrs = pd.read_csv(path)['PSNRS'].tolist()
          s = pd.read_csv(path)['SCORE'].tolist()
        else:
          psnrs = [0]*num_his
          s = [1]*num_his

        #for i in range(start, length):
        video_clip = data_loader.get_video_clips(video_name, start - num_his, start+1)
        psnr = sess.run(test_psnr_error,
                        feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
        psnrs.append(psnr)

        print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
            video_name, num_videos, start, length, psnr))
        if start==num_his:
          psnrs[0:num_his] = [psnr]*num_his
          s[0:num_his] = [1]*num_his
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
        
        print('SCORES: ', scores[-1])
        s.append(scores[-1])
        dfScore = pd.DataFrame(s, columns=['SCORE'])

        df = pd.concat([df, dfScore], axis=1)
        df.to_csv(path)
        if scores[-1]<0.73:
            return 'Anomaly' #print('Anomaly')
        else:
            return 'Normal'