#Creating the video of frames that is present currently with the test set
#call get_scores_label from evaluate.py into inference.py instead of compute_auc to get the output for the running frames
import os
import cv2

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item and '.mp4' not in item:
            result.append(item)
    return result


path = '/Users/iec/OneDrive/Ono Project/Anomaly Detection/Yash Ghatole (2021 Fall)/Cow Data/Data/AnomalyFrames/'
path2 = '/Users/iec/Documents/Anomaly Detection/Ono-Project/Data/cowData/testing/'
totalFolders = os.listdir(path)
totalFolders = ignoreDS_Store(totalFolders)

for folderName in totalFolders:
    videoName = folderName
    framePath = path+videoName+'/'
    videoNumbers = os.listdir(framePath)
    videoNumbers = ignoreDS_Store(videoNumbers)

    for videoNo in videoNumbers:
        #videoNo = '0'
        images = os.listdir(framePath+videoNo+'/')
        images = ignoreDS_Store(images)
        images = sorted(images, key=lambda x: int(x.split(".")[0]))
        img=[]
        for image in images:
            img.append(cv2.imread(framePath+videoNo+'/'+image))
            
        height,width,layers=img[1].shape
        video=cv2.VideoWriter(path2+videoName+'_'+videoNo+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20,(width,height))

        for j in range(len(img)):
            video.write(img[j])

        cv2.destroyAllWindows()
        video.release()