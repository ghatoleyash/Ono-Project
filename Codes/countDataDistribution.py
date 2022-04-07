from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd

path = '/Users/iec/Documents/Anomaly Detection/Ono-Project/Data/cowData/training/'
videoFolder = os.listdir(path)

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item and '.mp4' not in item:
            result.append(item)
    return result
videoFolder = ignoreDS_Store(videoFolder)

countNormalFrames = 0
for i in videoFolder:
    frames = os.listdir(path+i)
    frames = ignoreDS_Store(frames)
    countNormalFrames+=len(frames)

print("-----------------")
print(countNormalFrames)
print("-----------------")