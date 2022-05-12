import os 
from generatePred import generatePredictions
from GroundTruthAnnotate import GTAnnotate

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item and '.mp4' not in item:
            result.append(item)
    return result
#videoFolder = ignoreDS_Store(videoFolder)


videoRange = {}

path1 = '../Data/cowData/testing/'
pathCSV = '/Users/iec/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/Results/ResultCSV/'
videosToTest = os.listdir(path1)

for video in videosToTest:
    generatePredictions(video)
    filename = os.path.splitext(videoName)
    videoName = filename[0]
    ext = filename[1]
    
    #GTAnnotate(videoRange[video])
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
    
    print("Enter the range: ")
    rnge = input()
    GTAnnotate(rnge)
    predictionvsGroundtruth(1, video)

    os.replace('../Codes/PSNRS.csv', pathCSV+videoName+'.csv')
    os.rmdir(path1+'/frames')

