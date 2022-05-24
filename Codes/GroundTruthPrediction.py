"""
Implementation to annotate the video 
with ground truth(red cross) and 
prediction results with blue edge

Requires Ground Truth annotated PSNRS.csv to 
create annotation frame-by-frame
"""

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


def retrieveGTLabel(df, columns_list, count):
	try:
		compare = df.loc[df[columns_list[0]]==count, 'Ground Truth'].values[0]
		return compare
	except:
		return "error"


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
	columns_list = df.columns
	isExist = os.path.exists(checkResultPath)
	if not isExist:
		os.mkdir(checkResultPath)
	
	images = sorted(images, key=lambda x: int(x.split(".")[0]))
	img=[]
	count = 0
	label = 0
	for image in images:
		#print("IMAGES: ", image)
		image = cv2.imread(path+'/'+image)
		if check:
			compare = retrieveGTLabel(df, columns_list, count)
			if compare in [0,1]:
				if compare==0:#df['Ground Truth'].iloc[count] == 0:
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

	"""
	# CODE TO TRANSFER THE RESULTS FILE TO SOME OTHER FOLDER
	pathCSV = 'path/to/DesiredCSVFolder'
	pathVideo = 'path/to/DesiredVideo/Folder'
	pathFrames = '../Data/cowData/testing/frames/'
	os.replace('../Codes/PSNRS.csv', pathCSV+videoName+'Result.csv')
	os.replace(pathFrames+'ResultVideo/'+videoName+'Result.mp4', pathVideo+videoName+'Result.mp4')
	"""
	



#21	55-235
#32 20-270
#28 50-210


#25_0 9-36
#25_1 13-66
#25_2 12-17
#25_3 5-35
#25_4 11-31
#32_0 8-20
#32_1 5-53
#38_0 20-39
#38_1 18-32
#38_2 19-30
#45_0 13-28
#45_1 12-27
#45_2 8-36
#51_0 8-18
#58_0 8-14
#58_1 10-54
#06_0 9-20
#14_0 5-8
#14_1 9-19
#14_2 9-13
#14_3 11-25
#14_4 9-11
#21_0 13-46
#21_1 5-7
#21_2 9-20
#21_3 11-17
#28_0 10-41
#35_0 16-80
#35_1 13-25
#35_2 9-27
#58_21_0 9-12
#51EB_0 13-25
#51EB_1 12-18 
#59EB_0 12-36
#05EB_0 10-25
#13EB_0 6-23
#13EB_1 

