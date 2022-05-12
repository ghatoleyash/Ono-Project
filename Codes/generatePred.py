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
from initialization import initialization
from prediction import inference
from GroundTruthPrediction import predictionvsGroundtruth


def savingTheFrame(path, name, frame):
  cv2.imwrite(path+str(name)+'.jpg',frame)

def editingTheFrame(path, name, frame, result):
  if result=='Anomaly':
    start_point = (0, 0)
    end_point = (start_point[0]+len(frame[0]), start_point[1]+len(frame))
    color = (255, 255, 0)
    thickness = 20
    frame1 = frame.copy()
    frame1 = cv2.rectangle(frame1, start_point, end_point, color, thickness)
    savingTheFrame(path, name, frame1)
    #cv2.imwrite(path+str(name)+'.jpg',frame1)
    flag = 1
  else:
    savingTheFrame(path, name, frame)
    #cv2.imwrite(path+str(name)+'.jpg',frame)
  



#Name of the dataset
dataset = 'cowdata'

#def generatePredictions(video_name):
#Initializes the model
returnValues = initialization()

test_video_clips_tensor = returnValues[0]
test_inputs = returnValues[1]
test_gt = returnValues[2]
test_outputs = returnValues[3]
test_psnr_error = returnValues[4]

#
path1 = '../Data/'+dataset+'/testing/'
path2 = '../Data/'+dataset+'/testing/frames/01/'
path3 = '../Data/'+dataset+'/testing/frames/Results/'
isExistpath2 = os.path.exists(path2)
isExistpath3 = os.path.exists(path3)
if not isExistpath2:
  os.makedirs(path2)
if not isExistpath3:
  os.makedirs(path3)


print("Enter the video name with extension: (example.mp4)")
videoName = input()#video_name
print("Enter video name with extension: (example.mp4)--", videoName)

cap= cv2.VideoCapture(path1+videoName)
i=0
images = os.listdir('../Data/'+dataset+'/testing/frames/01')
while(cap.isOpened()):
    flag = 0
    ret, frame = cap.read()
    if ret == False:
        break
    
    #print('Images: ', len(images))
    name = i
    if i<=9:
      name = '0'+str(i)

    if i>=5: #and (len(images)%5==0):
      #call the model
      result = inference(test_video_clips_tensor, test_inputs, test_gt, test_outputs, test_psnr_error)
      print('Result: ', result)
      
      editingTheFrame(path3, name, frame, result)
      editingTheFrame(path2, name, frame, "Blank")

      print("INDEX: ", i)
    else:
      editingTheFrame(path2, name, frame, "Initial")
      editingTheFrame(path3, name, frame, "Initial")
      
    images = os.listdir('../Data/'+dataset+'/testing/frames/01')
    i+=1
    flag = 0
    # if i==10:
    #   break
cap.release()
cv2.destroyAllWindows()


# print("IMAGES LAST: ", len(images))
# print("COUNTER LAST: ", i)


#reading the PSNR.csv to add the Ground Truth Column
pathCSV = '../Codes/PSNRS.csv'
df = pd.read_csv(pathCSV)
df['Ground Truth'] = ""
df.to_csv(pathCSV)


while True:
  print("Press 1 Comparison with GroundTruth else 0")
  check = input()
  if check in ["0","1"]:
    break
check = int(check)
if check:
  print("Once finish with the annotations press ctrl+c ")
  while True:
    try:
      time.sleep(1)
    except KeyboardInterrupt:
      break
  
predictionvsGroundtruth(check, videoName)

