"""
Convert the videos to their respective Frames
"""

import os 
import numpy as np
import cv2
import pandas as pd
import time

#Path to the Videos
path1 = '/Users/iec/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/Results/10sec_Results/10sec_RGB_Videos/'
#Path to save the Frames from the Videos
path2 = '/Users/iec/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/Results/10sec_Results/10sec_BW_Frames/'


speed = 1 #how to sample every 2 frame (if given 5-every 5 frame is saved)

videos = os.listdir(path1)
videoSet = set()

#There are repeated videos for same time ending with "(1)""
for video in videos:
  if '(1)' not in video:
    videoSet.add(video)

for video in videoSet:
  print('Video Name', video)
  fNameExt = os.path.splitext(video)
  fileName = fNameExt[0]
  fileExt = fNameExt[1]

  framePath = path2 + fileName
  print(framePath)
  isExist = os.path.exists(framePath)
  if not isExist:
    os.makedirs(framePath)

  cap= cv2.VideoCapture(path1+video)
  i=0
  flag = 0
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      images = os.listdir(framePath+'/')
      if i%speed==0:
        name = str(i)
        if i<=9:
            name = '0'+str(i)
        
        #grayscale image save
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(framePath+'/'+name+'.jpg',grayImage)
      i+=1
      flag = 0

  cap.release()
  cv2.destroyAllWindows()
  print("In Sleep waiting for upload")
  time.sleep(10)
