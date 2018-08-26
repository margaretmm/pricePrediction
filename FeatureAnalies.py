import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def anova(frame, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        #print(c)
        #print(frame[c].unique())
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['Price'].values
            samples.append(s)  # 某特征下不同取值对应的房价组合形成二维列表
        pval = stats.f_oneway(*samples)[1]  # 一元方差分析得到 F，P，要的是 P，P越小，对方差的影响越大。
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

def encode(frame, feature):
    '''
    对所有类型变量，依照各个类型变量的不同取值对应的样本集内房价的均值，按照房价均值高低
    对此变量的当前取值确定其相对数值1,2,3,4等等，相当于对类型变量赋值使其成为连续变量。
    此方法采用了与One-Hot编码不同的方法来处理离散数据，值得学习
    注意：此函数会直接在原frame的DataFrame内创建新的一列来存放feature编码后的值。
    '''
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, 'Price']].groupby(feature).mean()['Price']
    # 上述 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]: {'Grvl': 1, 'MISSING': 3, 'Pave': 2}
        frame.loc[frame[feature] == attr_v, feature+'_E'] = score

dfName='Data_washed.csv'
df_ori = pd.read_csv(dfName,header=0)
df=df_ori[df_ori['Area']>50]# filter non normal house
df_train=df[["Rooms","Area","Decorate","subway","FiveYear","hasLift","Toward_s","Toward_n","Toward_e","Toward_w","Floor_h","Floor_m","Floor_l"]]
df_target=df["Price"].values.reshape(len(df),1)
print(np.shape(df_train))
print(np.shape(df_target))


quantity = [attr for attr in df_train.columns if df_train.dtypes[attr] != 'object']  # 数值变量集合
quality = [attr for attr in df_train.columns if df_train.dtypes[attr] == 'object']  # 类型变量集合
for c in quality:  # 类型变量缺失值补全
    df_train[c] = df_train[c].astype('category')
    if df_train[c].isnull().any():
        df_train[c] = df_train[c].cat.add_categories(['MISSING'])
        df_train[c] = df_train[c].fillna('MISSING')

# 连续变量缺失值补全
quantity_miss_cal = df_train[quantity].isnull().sum().sort_values(ascending=False)  # 缺失量均在总数据量的10%以下
missing_cols = quantity_miss_cal[quantity_miss_cal>0].index
df_train[missing_cols] = df_train[missing_cols].fillna(0.)  # 从这些变量的意义来看，缺失值很可能是取 0
df_train[missing_cols].isnull().sum()  # 验证缺失值是否都已补全

a = anova(df,quantity)
a['disparity'] = np.log(1./a['pval'].values)  # 悬殊度
fig, ax = plt.subplots(figsize=(16,8))
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)
plt.show()


# quality_encoded = []
# # 由于qualitative集合中包含了非数值型变量和伪数值型变量（多为评分、等级等，其取值为1,2,3,4等等）两类
# # 因此只需要对非数值型变量进行encode()处理。
# # 如果采用One-Hot编码，则整个qualitative的特征都要进行pd,get_dummies()处理
# for q in quantity:
#     encode(df_ori, q)
#     quality_encoded.append(q+'_E')
# df_ori.drop(quantity, axis=1, inplace=True)  # 离散变量已经有了编码后的新变量，因此删去原变量
# # df_tr.shape = (1460, 80)
# print(quality_encoded, '\n{} qualitative attributes have been encoded.'.format(len(quality_encoded)))

def spearman(frame, features):
    '''
    采用“斯皮尔曼等级相关”来计算变量与房价的相关性(可查阅百科)
    此相关系数简单来说，可以对上述encoder()处理后的等级变量及其它与房价的相关性进行更好的评价（特别是对于非线性关系）
    '''
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['Price'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.show()
features = quantity
spearman(df_ori, features)


