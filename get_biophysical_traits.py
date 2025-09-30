import pandas as pd
import numpy as np
from datetime import date
import seaborn as sns
from matplotlib import pyplot as plt

folder = rf'C:\Users\mqalborn\Desktop\GRAPEX\RIP\outcomes/LAI2200_B_FOOTPRINT_OV_UAVS_STATS.csv'
data_bt = pd.read_csv(folder).drop('Unnamed: 0', axis=1)
data_bt = data_bt[data_bt.B==1]
data_bt = data_bt.astype({'block': str, 'transect': str, 'tree': str, 'B': str})

data_bt = data_bt[['index', 'block', 'transect', 'tree', 'timestamp',
                   'tree_area', 'Fov', 'Fun', 'Fsoil', 'Hc', 'TIR_mean']]

data_lai = pd.read_csv(rf'C:\Users\mqalborn\Desktop\GRAPEX\RIP\LAI/20190504_RIP_720_LAI_summary.csv')
data_lai = data_lai.astype({'block': str, 'transect': str, 'tree': str})

data_lai = data_lai[['block', 'transect', 'tree', 'LAI_ov', 'ACF_ov', 'lat_ov', 'lon_ov', 'date_LAI_un', 'LAI_un', 'ACF_un']]

data_bt = pd.merge(data_bt, data_lai, on=['block', 'transect', 'tree'])
data_bt = data_bt.drop(['lat_ov', 'lon_ov', 'date_LAI_un'], axis=1)

data_bt = data_bt.rename(columns={'timestamp': 'timestamp_uavs'})
data_bt.timestamp_uavs = '20190504'
data_bt.loc[:, 'date'] = date(2019, 8, 15)

data_bt.loc[:, 'Wov'] = np.sqrt(data_bt.tree_area / np.pi) * 2



path = rf'C:\Users\mqalborn\Box\Crop_Sensing_Projects\GRAPEX\ground\RIP\MCM\RIP_722\PROCESSED\HOURLY\UNCLEAN\2019/RIP_722_CSI_subdaily_2019.csv'
data_ec = pd.read_csv(path)
data_ec = data_ec[['TIMESTAMP', 'Site', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT']]
data_ec = data_ec.rename(columns={'TIMESTAMP': 'timestamp_ec'})
data_ec.TIMESTAMP = pd.to_datetime(data_ec.timestamp_ec, format='%m/%d/%Y %H:%M')
data_ec.loc[:, 'date'] = pd.to_datetime(data_ec.timestamp_ec, format='%m/%d/%Y %H:%M').dt.date

data_ec.loc[:, 'hour'] = (pd.to_datetime(data_ec.timestamp_ec, format='%m/%d/%Y %H:%M').dt.hour +
                          pd.to_datetime(data_ec.timestamp_ec, format='%m/%d/%Y %H:%M').dt.minute / 60)
data_ec = data_ec[data_ec.date == date(2019, 8, 15)]

data = pd.merge(data_bt, data_ec, on='date')
data.to_csv(rf'C:\Users\mqalborn\Desktop\GRAPEX\RIP\outcomes/biophysical_traits.csv', index=False)
# sns.scatterplot(data_ec, x='hour', y='SW_IN')
# plt.show()
# print(data_ec)