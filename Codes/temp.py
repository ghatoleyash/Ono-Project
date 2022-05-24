import pandas as pd
import numpy as np

def newPSNRCompute(x, minPSNR, maxPSNR):
    newPSNR = (x - minPSNR)/(maxPSNR-minPSNR)
    return newPSNR


pathCSV = '/Users/iec/Documents/Sampled Test Set/CSV/2021-09-28_09-06-21_SampledResult.csv'
#path2 = '../Data/'+dataset+'/testing/frames/'

df = pd.read_csv(pathCSV)
columnsList = df.columns
maxPSNR = max(df['PSNRS'])
minPSNR = min(df['PSNRS'])
print('-----------MAX/MIN------------')
print(maxPSNR)
print(minPSNR)

df['newSCORE'] = 0
df['newSCORE'] = df['PSNRS'].apply(newPSNRCompute, args = [minPSNR, maxPSNR])
print('-----------RESULT------------')

df['diff'] = df['SCORE']-df['newSCORE']
print(df)

print(df.loc[df['diff']<-0.1])

#df.to_csv('/Users/iec/Documents/Sampled Test Set/CSV/2021-09-28_09-06-21_SampledResult.csv', index=False)
