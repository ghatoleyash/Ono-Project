"""
Creating the video of frames 
"""

import os
import cv2

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item and '.mp4' not in item:
            result.append(item)
    return result

#Path where frame exist
path = '/Users/iec/Documents/Sampled Test Set/Frames/'

#Path where Video will be stored from the above frames
path2 = '/Users/iec/Documents/Sampled Test Set/Video/'

totalFolders = os.listdir(path)
totalFolders = ignoreDS_Store(totalFolders)

for folderName in totalFolders:
    videoName = folderName
    print(videoName)
    framePath = path+videoName+'/'

    images = os.listdir(framePath+'/')
    images = ignoreDS_Store(images)
    images = sorted(images, key=lambda x: int(x.split(".")[0]))
    img=[]
    for image in images:
        img.append(cv2.imread(framePath+'/'+image))
        
    height,width,layers=img[1].shape
    video=cv2.VideoWriter(path2+videoName+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20,(width,height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()