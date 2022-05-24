"""
Creating the summary statistics, 
for a single video such as total number of frames,
total number of normal frames, total number of
anomalous frames within the video
"""

import pandas as pd
import numpy as np
import os
import collections
#Columns = videoname, total frames, normal frames, anomalous frames

pathVideoCSV = '/Users/iec/Documents/Data/AnomalyCSV'
pathNormalFrames = '/Users/iec/Documents/Data/NormalFrames'
pathAnomalyFrames = '/Users/iec/Documents/Data/AnomalyFrames'


videoDic = collections.defaultdict(list)

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item and '.mp4' not in item:
            result.append(item)
    return result


def calculateSummaryStats():
    df = pd.read_excel(open(pathVideoCSV+'/AnomalyLabel.xlsx', 'rb'), sheet_name='Anomaly') 

    uniqueVideos = df['video'].unique()

    for video in uniqueVideos:
        countNormalFrames = os.listdir(pathNormalFrames+'/'+video)
        countNormalFrames = len(ignoreDS_Store(countNormalFrames))
        
        AnomalyNumbers = os.listdir(pathAnomalyFrames+'/'+video) 
        AnomalyNumbers = ignoreDS_Store(AnomalyNumbers)
        countAnomalousFrames = 0
        for n in AnomalyNumbers:
            anomalousFrames = os.listdir(pathAnomalyFrames+'/'+video+'/'+n)
            anomalousFrames = ignoreDS_Store(anomalousFrames)
            countAnomalousFrames+=len(anomalousFrames)

        totalFrames = countNormalFrames + countAnomalousFrames
        videoDic[video] = [totalFrames, countNormalFrames, countAnomalousFrames]

    videoNames = videoDic.keys()
    videoSummarystats = videoDic.values()
    df = pd.DataFrame(videoSummarystats, columns = ['TotalFrames', 'NormalFrames', 'AnomalousFrames'])
    df['VideoName'] = videoNames
    df = df[['VideoName', 'TotalFrames', 'NormalFrames', 'AnomalousFrames']]
    df.to_csv(pathVideoCSV+'/videoSummaryStats.csv')

calculateSummaryStats()




