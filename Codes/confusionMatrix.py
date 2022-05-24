"""
Creating the confusion matrix for test set
"""

from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd

threshold = 0.73
path = '/Users/iec/OneDrive/Yash Ghatole (2021 Fall)/Cow Data/Results/ResultCSV/'
csvFiles = os.listdir(path)
df = pd.DataFrame()

def checkThreshold(score, threshold):
    if score<=threshold:
        return 0
    return 1

def ignoreDS_Store(lst):
    result = []
    for item in lst:
        if (".DS_Store" in item) or (".mp4" in item) or ("PSNRS" in item) or (".py" in item):
            continue
        else:
            print(item)
            result.append(item)
    print("RESULTS -------- ", len(result), "VIDEOS")
    return result


csvFiles = ignoreDS_Store(csvFiles)
#print(csvFiles)
for csv in csvFiles:
    dfTemp = pd.read_csv(path+csv)
    dfTemp['Predicted'] = dfTemp.apply(lambda row: int(checkThreshold(row['SCORE'], threshold)), axis=1)
    df = df.append(dfTemp)
df['Ground Truth'] = df['Ground Truth'].astype('int')

y_pred = df['Predicted']
y_actual = df['Ground Truth']


print('0-ANOMALY')
print('1-NORMAL')
print('-------------------------CONFUSION MATRIX-------------------------------')
print(confusion_matrix(y_actual, y_pred))
tp, fn, fp, tn = confusion_matrix(y_actual, y_pred).ravel()
print("True Positives: ", tp)
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("Total Frames: ", tp+tn+fp+fn)
print('------------------------------------------------------------------------')
print()
print('-------------------------CLASSIFICATION MATRIX--------------------------')
print(classification_report(y_actual, y_pred, labels=[0,1]))
print('------------------------------------------------------------------------')


