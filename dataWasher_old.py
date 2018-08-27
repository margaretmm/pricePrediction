import math

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
col = ['Address', 'Type','Area', 'Towards', 'Floor', 'Decorate', 'Feature', 'TotalPrice', 'Price']
dict={"":"2室2厅2卫","":"3室2厅2卫","":"4室2厅2卫","":"1室","":"1室1厅1卫","":"1室2厅2卫"}
age=['g1','g2','g3','g4']#<3,<5,6-10,>10
pd.options.display.max_rows = 10
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.1f}'.format

# 加载数据集
df = pd.read_csv("Data_test.csv", sep=',')
df.drop_duplicates()
grade_split = pd.DataFrame((x.split('_') for x in df.Feature),index=df.index,columns=['subway','5Years','Haslift'])
df=pd.merge(df,grade_split,right_index=True, left_index=True)

df=df.fillna(0)
# print(df['subway'].isnull().value_counts())
df.insert(1, 'g1', 1)
df.insert(2, 'south', 0)
df.insert(3, 'north', 0)
df.insert(4, 'east', 0)
df.insert(5, 'west', 0)

df['south']=[1 if "南" in x[0] else 0 for x in df.Towards]
df['north']=[1 if "北" in x[0] else 0 for x in df.Towards]
df['east']=[1 if "东" in x[0] else 0 for x in df.Towards]
df['west']=[1 if "西" in x[0] else 0 for x in df.Towards]


df.insert(6, 'FloorHigh', 0)
df.insert(7, 'FloorMid', 0)
df.insert(8, 'FloorLow', 0)
df['FloorHigh']=[1 if "高" in x else 0 for x in df.Floor]
df['FloorMid']=[1 if "中" in x else 0 for x in df.Floor]
df['FloorLow']=[1 if "低" in x else 0 for x in df.Floor]

df['subway']=[1 if x and "距离" in x else 0 for x in df.subway]
df['5Years']=[1 if x and "满五" in x else 0 for x in df['5Years']]
df['Haslift']=[1 if x and "电梯" in x else 0 for x in df.Haslift]
df['Decorate']=[1 if x and "装修" in x else 0 for x in df.Decorate]

# df['TotalPrice']=df['TotalPrice'].apply(lambda x: x /100)
# df['Price']=df['Price'].apply(lambda x: x /10000)
# df['Area']=df['Area'].apply(lambda x: x /10)

del df['Address']
del df['Towards']
del df['Feature']
del df['Floor']
#df['Decorate']=[1 if "装修" in df['Decorate'][0] else 0 for x in df.Towards]
print(df.head(8) )
df.to_csv('Data_washed.csv',index=False,sep=',')
#df_bj_g1['rooms']=np.array(1 if df_bj_g1['Type'].index("1室")>-1 else 2 if  df_bj_g1['Type'].index("2室")>-1 else 3 if df_bj_g1['Type'].index("3室")>-1 else 4 if df_bj_g1['Type'].index("4室")>-1 ).astype(np.int32)
