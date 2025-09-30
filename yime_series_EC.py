import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

blocks = ['RIP_721', 'RIP_722', 'RIP_723', 'RIP_724']
path = r'C:\Users\mqalborn\Desktop\GRAPEX\RIP\FLUX\{}\PARTITIONING/*.csv'
dir_path = [path.format(x) for x in blocks]
files_path = [glob.glob(x) for x in dir_path]
files_path = [item for sublist in files_path for item in sublist]

print(files_path)
def read_csv(x):
    data = pd.read_csv(x)
    data.loc[:, 'block'] = x.split('\\')[-3]
    return data

data = [read_csv(x) for x in files_path]
data = pd.concat(data)
data = data.rename(columns={'Unnamed: 0': 'date'})
data.loc[:, 'timestamp'] = pd.to_datetime(data.date)
data.loc[:, 'doy'] = data.timestamp.dt.day_of_year
data.loc[:, 'year'] = data.timestamp.dt.year
data = data.groupby(['doy', 'year', 'block']).agg(LE=('LE', 'mean')).reset_index()

print(data.head())
print(data.columns)
# print(data['Unnamed: 0'])
# plt.figure(figsize=[10, 6])
sns.relplot(data=data, x='doy', y='LE', hue='block',
            row='year', aspect=2)
plt.show()