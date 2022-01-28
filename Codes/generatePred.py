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
from prediction import inference


path1 = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/'
path2 = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/01/'
path3 = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/Results/'
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
    images = os.listdir('/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/01')
    print('Images: ', len(images))
    if len(images)>=5: #and (len(images)%5==0):
      #call the model
      result = inference()
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