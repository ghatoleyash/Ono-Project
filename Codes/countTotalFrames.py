import pandas as pd
import os


def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if ".DS_Store" not in item:
            if '.py' not in item:
                if 'PSNRS' not in item:
                    result.append(item)
    return result

path = '/Users/iec/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/Results/ResultCSV/'
allFiles = os.listdir(path)
filteredCSV = ignoreDS_Store(allFiles)

print(filteredCSV)

countTotal = 0
for f in filteredCSV:
    df = pd.read_csv(path+f, index_col = False)
    countTotal+=df.shape[0]
    print('DATAFRAME SHAPE: ', df.shape)
print('------------TOTAL NUMBER OF VIDEOS: ' + str(len(filteredCSV)) + ' ---------------')
print('------------COUNT TOTAL: ' + str(countTotal) + ' ---------------')