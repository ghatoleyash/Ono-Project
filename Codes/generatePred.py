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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os 
import numpy as np
import cv2
import pandas as pd
from initialization import initialization
from prediction import inference
from GroundTruthPrediction import predictionvsGroundtruth

"""
returnValues = initialization()

test_video_clips_tensor = returnValues[0]
test_inputs = returnValues[1]
test_gt = returnValues[2]
test_outputs = returnValues[3]
test_psnr_error = returnValues[4]

path1 = '../Data/ped1/testing/'
path2 = '../Data/ped1/testing/frames/01/'
path3 = '../Data/ped1/testing/frames/Results/'
isExist = os.path.exists(path2)
if not isExist:
  os.makedirs(path2)


cap= cv2.VideoCapture(path1+'video_02.mp4')
i=0
flag = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    images = os.listdir('../Data/ped1/testing/frames/01')
    print('Images: ', len(images))
    if len(images)>=5: #and (len(images)%5==0):
      #call the model
      result = inference(test_video_clips_tensor, test_inputs, test_gt, test_outputs, test_psnr_error)
      #print('Result: ', result)
      
      if result=='Anomaly':
        start_point = (0, 0)
        end_point = (start_point[0]+len(frame[0]), start_point[1]+len(frame))
        color = (255, 255, 0)
        thickness = 2
        frame1 = frame.copy()
        frame1 = cv2.rectangle(frame1, start_point, end_point, color, thickness)
        cv2.imwrite(path3+str(name)+'.jpg',frame1)
        flag = 1
	

      #get the predictions
      #Annotate and save it in different folder with sequential names
      #save the next frame
      name = i
      if i<=9:
        name = '0'+str(i)
      cv2.imwrite(path2+str(name)+'.jpg',frame)
      if not flag:
        cv2.imwrite(path3+str(name)+'.jpg',frame)
     
    else:
      name = i
      if i<=9:
        name = '0'+str(i)
      cv2.imwrite(path2+name+'.jpg',frame)
      cv2.imwrite(path3+name+'.jpg',frame)
    i+=1
    flag = 0

cap.release()
cv2.destroyAllWindows()
"""

print("Once finish with the annotations press ctrl+c ")
while True:
	try:
		time.sleep(1)
	except KeyboardInterrupt:
		break
		

predictionvsGroundtruth()

