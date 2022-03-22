import os 
import numpy as np
import cv2
import pandas as pd
import time
import collections


def anomalyFrameListFunc(fps, fileName, df, rows, columns):
    #list_of_list start-end
    frameDict = collections.defaultdict(int)
    frameList = []
    for i in range(rows):
        # print('START TIME')
        # print(str(df.iloc[i]['start_time']).split(':'))
        mmStart, ssStart, t = str(df.iloc[i]['start_time']).split(':')
        mmEnd, ssEnd, t = str(df.iloc[i]['end_time']).split(':')
        mmStart, ssStart, t = int(mmStart), int(ssStart), int(t)
        mmEnd, ssEnd, t = int(mmEnd), int(ssEnd), int(t)

        fStart = fps*(mmStart*60+ssStart)
        fEnd = fps*(mmEnd*60+ssEnd)
        frameDict[fStart] = fEnd
        frameList.append([fStart, fEnd])

    return frameDict, frameList



#onedrive path
anomalycsvpath = '/Users/iec/Documents/Data/AnomalyCSV/AnomalyLabel.xlsx'
#'/home/iec/Yash/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/AnomalyLabel.xlsx'

#onedrive path
AnomalyPath = '/Users/iec/Documents/Data/AnomalyFrames/'
#'/home/iec/Yash/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/AnomalyFolder/'

#onedrive path
path1 = '/Users/iec/Documents/Data/Video/'
#'/home/iec/Yash/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/ManyCows/20210928AM/'

#onedrive path
path2 = '/Users/iec/Documents/Data/NormalFrames/'
#'/home/iec/Yash/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/ManyCows/FrameData/'

#path2 = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/01/'
#path3 = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/Results/'
#/content/drive/MyDrive/20210928AM


#Read anomaly csv file
dfAnomaly = pd.read_excel(anomalycsvpath, sheet_name='Anomaly', engine='openpyxl')

speed = 5 #how to sample every 2 frame (if given 5-every 5 frame is saved)

videos = os.listdir(path1)
videoSet = set()
#os.path.splitext

#There are repeated videos for same time ending with "(1)""
for video in videos:
    if '(1)' not in video:
        videoSet.add(video)
#print(len(videos)-len(videoSet))

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


    #find if the video contains anomaly (entry exist in anomalyLabel.xlsx)
    dfTemp = dfAnomaly.loc[dfAnomaly['video']==fileName]
    rows, columns = dfTemp.shape
    if rows==0:
        continue
    # print('DATAFRAME')
    # print(dfTemp)

    #save anomalous video in different folder    
    #create folder with anomaly video
    #if there are multiple anomaly within a single video number them and save accordinly
    cap= cv2.VideoCapture(path1+video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    anomalyframeDict, anomalyframeList = anomalyFrameListFunc(fps, fileName, dfTemp, rows, columns)
    
    print('FRAME LIST: ', anomalyframeList)
    i=0
    flag = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        images = os.listdir(framePath+'/')#'/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Data/ped1/testing/frames/01')
        #print('Images: ', len(images))
        anomalousFrame = 0
        indx = 0
        for j in range(len(anomalyframeList)):
            #print('FRAME', fr)
            fr = anomalyframeList[j]
            if fr[0]<=i and i<=fr[1]:
                anomalousFrame = 1
                indx = j
                break 

        if anomalousFrame:
            anomalyVideoNumber = AnomalyPath+fileName+'/'+str(indx)
            isanomalyVidNoExist = os.path.exists(anomalyVideoNumber)
            if not isanomalyVidNoExist:
                os.makedirs(anomalyVideoNumber)

            if i%speed==0:
                name = str(i)
                if i<=9:
                    name = '0'+str(i)
                
                #grayscale image save
                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(anomalyVideoNumber+'/'+name+'.jpg',grayImage)

        else:
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
    # print("In Sleep waiting for upload")
    # time.sleep(10*60)


