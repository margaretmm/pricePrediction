import numpy as np
import pandas as pd

dfName='Data_binjiang_g1.csv'
df = pd.read_csv(dfName)


col=['Address','Rooms', 'Area', 'Towards', 'Floor', 'Decorate','Feature', 'TotalPrice', 'Price']

del df['Address']

#df['Rooms']=[1 if "1室" in df['Rooms']]

df.fillna(0,inplace=True)


# a = df['Feature'].str.split('_',3,expand=True)#, .apply(pd.Series))
# print(a)

#df2=pd.DataFrame([df['Feature'].str.split('_',expand=True)])
# newDF=df['Feature'].str.split('_',3, expand=True)#.apply(pd.value_counts).fillna(0).astype(str).reset_index()
#df2.columns = ['subway', 'FiveYear','hasLift']
#df2=pd.merge(df,newDF,on='Feature')
def roomChange(x):
    if "1室" in x:
        return 1
    elif  "2室" in x:
        return 2
    elif  "3室" in x:
        return 3
    elif  "4室" in x:
        return 4
    elif  "5室" in x:
        return 5
    elif  "6室" in x:
        return 6
    else:
        return 10

def TowardChange(x):
    if x.contains("南"):
        return 1
#print(df['Towards'].str.index("南")>-1)
df['Rooms']=df['Rooms'].apply(lambda x: roomChange(x))
df['Toward_s']=np.array(df['Towards'].str.contains("南")).astype(np.int)
df['Toward_n']=np.array(df['Towards'].str.contains("北")).astype(np.int)
df['Toward_e']=np.array(df['Towards'].str.contains("东")).astype(np.int)
df['Toward_w']=np.array(df['Towards'].str.contains("西")).astype(np.int)
del df['Towards']


df['Floor_h']=np.array(df['Floor'].str.contains("高")).astype(np.int)
df['Floor_m']=np.array(df['Floor'].str.contains("中")).astype(np.int)
df['Floor_l']=np.array(df['Floor'].str.contains("低")).astype(np.int)
del df['Floor']

df['Decorate']=np.array(df['Decorate'].str.contains("装修").astype(np.int))

#print("adfg".__contains__("x"))
def contain(x,subStr):
    print(x)
    if x==0 or x=='0':
        return 0
    elif x.__contains__(subStr):
        return 1
    else:
        return 0


df['subway']=df['subway'].apply(lambda x:contain(x,"米"))
df['FiveYear']=df['FiveYear'].apply(lambda x:contain(x,"满五"))
df['hasLift']=df['hasLift'].apply(lambda x:contain(x,"电梯"))
#df['Area']=[df['Area'].str.extract(r'(\d)')]
print(df.head(15))


df['BuyYesrs<3']=[1 if dfName.__contains__("g1") else 0]
df['BuyYesrs_3_5']=[1 if dfName.__contains__("g2") else 0]
df['BuyYesrs_6_10']=[1 if dfName.__contains__("g3") else 0]
df['BuyYesrs>10']=[1 if dfName.__contains__("g4") else 0]
df.to_csv('Data_washed.csv',sep=',',index=None)