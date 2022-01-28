import pandas as pd

#Human Annotation
path = '/content/drive/MyDrive/IECLab/Experiments/Reconstruction Error/ano_pred_cvpr2018/Codes/PSNRS.csv'
df = pd.read_csv(path)
df['Ground Truth'] = 0
df.to_csv('PSNRS.csv')

print('Press q whenever finished with human annotation')
# while True:
#     print()
