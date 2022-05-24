"""
Short code to annotate the frames within 
PSNRS.csv file if the anomaly start and 
end frame number is known without the
manual adding of Ground Truth labels
"""

import pandas as pd

#Enter range 
print("Enter Range (e.g: 5-10)")
rangeInpt = input()
start, end = rangeInpt.split("-")
start, end = int(start), int(end)
path = '../Codes/PSNRS.csv'
df = pd.read_csv(path, index_col = False)
df[['Ground Truth']] = 1
df.loc[start:end,['Ground Truth']] = 0
df.to_csv('../Codes/PSNRS.csv', index = False)


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
#59EB_0 
#05EB_0 
#13EB_0 
#13EB_1 

