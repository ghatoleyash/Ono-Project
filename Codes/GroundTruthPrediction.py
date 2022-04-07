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

def predictionvsGroundtruth(check, videoName):
	dataset = 'cowdata'
	path = '../Data/'+dataset+'/testing/frames/Results/'
	path1 = '../Codes/PSNRS.csv'
	path2 = '../Data/'+dataset+'/testing/frames/'
	videoResultDir = 'ResultVideo'
	
	filename = os.path.splitext(videoName)
	videoName = filename[0]
	ext = filename[1]
	
	images = os.listdir(path)
	checkResultPath = os.path.join(path2, videoResultDir)
	df = pd.read_csv(path1)
	isExist = os.path.exists(checkResultPath)
	if not isExist:
		os.mkdir(checkResultPath)
	
	images = sorted(images, key=lambda x: int(x.split(".")[0]))
	#print(images)
	img=[]
	count = 0
	label = 0
	for image in images:
	  image = cv2.imread(path+'/'+image)
	  if check:
		  if df['Ground Truth'].iloc[count] == 0:
		    # Getting the height and width of the image
		    height = image.shape[0]
		    width = image.shape[1]
		    # Drawing the lines
		    cv2.line(image, (0, 0), (width, height), (0, 0, 255), 5)
		    cv2.line(image, (width, 0), (0, height), (0, 0, 255), 5)

	  img.append(image)
	  count+=1
	    
	height,width,layers=img[1].shape
	#DIVX
	video=cv2.VideoWriter(checkResultPath+'/'+videoName+'Result'+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20,(width,height))
	#video=cv2.VideoWriter('../Data/ped1/testing/video_Result_02_Annotated_term.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20,(width,height))

	for j in range(len(img)):
	  video.write(img[j])

	cv2.destroyAllWindows()
	video.release()
