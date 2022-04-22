import pandas as pd

print("Enter the range")
rangeInpt = input()
start, end = rangeInpt.split("-")
start, end = int(start), int(end)
path = '../Codes/PSNRS.csv'
df = pd.read_csv(path, index_col = False)
df[['Ground Truth']] = 1
df.loc[start:end,['Ground Truth']] = 0
df.to_csv('../Codes/PSNRS.csv', index = False)


