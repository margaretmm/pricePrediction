import numpy as np
import pandas as pd
import re


def roomChange(x):
    if "1室" in x:
        return 1
    elif "2室" in x:
        return 2
    elif "3室" in x:
        return 3
    elif "4室" in x:
        return 4
    elif "5室" in x:
        return 5
    elif "6室" in x:
        return 6
    else:
        return 10


def TowardChange(x):
    if x.contains("南"):
        return 1

def Wash(area='bingjiang', index=0):
    dfName='Data_'+area+'_g'+str(index)+'.csv'
    df = pd.read_csv(dfName, sep=',')
    pd.set_option('max_colwidth',100)
    pd.set_option('display.max_columns', None)
    # if df.empty():
    #     print("read_csv from " +dfName+" have no data!!!")
    #     exit(1)

    del df['Address']
    col=['Address','Rooms', 'Area', 'Towards', 'Floor', 'Decorate','Feature', 'TotalPrice', 'Price']
    df=df.fillna(0)



    #print(df['Towards'].str.index("南")>-1)
    df['Rooms']=df['Rooms'].apply(lambda x: roomChange(x))
    df['Decorate']=np.array(df['Decorate'].str.contains("装修").astype(np.int))
    #df['Decorate']=[1 if x and "装修" in x else 0 for x in df.Decorate]

    #print(df.head(3))
    cols = list(df)
    cols.insert(2, cols.pop(cols.index('Decorate')))
    df = df.ix[:, cols]
    df.insert(3, 'subway', 0)
    df.insert(4, 'FiveYear', 0)
    df.insert(5, 'hasLift', 0)
    #print(df.head(3))

    # for x in df.Feature:
    #     print(x)
    #     print( "距离" in x)
    df['subway']=[1 if x!=0 and "距离" in x else 0 for x in df.Feature]
    df['FiveYear']=[1 if  x!=0 and "满五" in x else 0 for x in df.Feature]
    df['hasLift']=[1 if  x!=0 and "电梯" in x else 0 for x in df.Feature]
    del df["Feature"]

    df['Toward_s']=np.array(df['Towards'].str.contains("南")).astype(np.int)
    df['Toward_n']=np.array(df['Towards'].str.contains("北")).astype(np.int)
    df['Toward_e']=np.array(df['Towards'].str.contains("东")).astype(np.int)
    df['Toward_w']=np.array(df['Towards'].str.contains("西")).astype(np.int)
    del df['Towards']

    df['Floor_h']=np.array(df['Floor'].str.contains("高")).astype(np.int)
    df['Floor_m']=np.array(df['Floor'].str.contains("中")).astype(np.int)
    df['Floor_l']=np.array(df['Floor'].str.contains("低")).astype(np.int)
    del df['Floor']


    #print(dfName.__contains__("g1"))
    if dfName.__contains__("g1"):
        df['BuyYesrs<3']=np.array(1).astype(np.int)
        df['BuyYesrs_3_5']=np.array(0).astype(np.int)
        df['BuyYesrs_6_10']=np.array(0).astype(np.int)
        df['BuyYesrs>10']=np.array(0).astype(np.int)
    elif  dfName.__contains__("g2"):
        df['BuyYesrs<3']=np.array(0).astype(np.int)
        df['BuyYesrs_3_5']=np.array(1).astype(np.int)
        df['BuyYesrs_6_10']=np.array(0).astype(np.int)
        df['BuyYesrs>10']=np.array(0).astype(np.int)
    elif  dfName.__contains__("g3"):
        df['BuyYesrs<3']=np.array(0).astype(np.int)
        df['BuyYesrs_3_5']=np.array(0).astype(np.int)
        df['BuyYesrs_6_10']=np.array(1).astype(np.int)
        df['BuyYesrs>10']=np.array(0).astype(np.int)
    elif  dfName.__contains__("g4"):
        df['BuyYesrs<3']=np.array(0).astype(np.int)
        df['BuyYesrs_3_5']=np.array(0).astype(np.int)
        df['BuyYesrs_6_10']=np.array(0).astype(np.int)
        df['BuyYesrs>10']=np.array(1).astype(np.int)
    #print(df.head(15))

    # for x in df.Price:
    #     print(x)
    #     a=re.findall(r"^(\d+)", x)
    #     print(a[0], type(a))

    df['Price']=[re.findall(r"^(\d+)", x)[0] if u"元/㎡" in x else 0 for x in df.Price]
    df['Price'] = df['Price'].astype(np.float64)
    df = df[(df['Price']>5000) &(df['Price']<80000)]

    df['TotalPrice']=[re.findall(r"^(\d+)", x)[0] if u"万元" in x else 0 for x in df.TotalPrice]
    df['TotalPrice'] = df['TotalPrice'].astype(np.float32)
    df = df[(df['TotalPrice']>60) &(df['TotalPrice']<1800)]

    df['Area']=[re.findall(r"^(\d+)", x)[0] if u"㎡" in x else 0 for x in df.Area]
    df['Area'] = df['Area'].astype(np.float32)
    df.drop_duplicates()

    df['ProductHouse']=np.array(df['Area']>50).astype(np.int)

    df['area_bj']=np.array(0).astype(np.int)
    df['area_gs']=np.array(0).astype(np.int)
    df['area_yh']=np.array(0).astype(np.int)
    df['area_xh']=np.array(0).astype(np.int)
    df['area_xc']=np.array(0).astype(np.int)
    df['area_xs']=np.array(0).astype(np.int)
    if dfName.__contains__("binjiang"):
        df['area_bj']=np.array(1).astype(np.int)
    elif  dfName.__contains__("gongshu"):
        df['area_gs']=np.array(1).astype(np.int)
    elif  dfName.__contains__("xihu"):
        df['area_xh']=np.array(1).astype(np.int)
    elif  dfName.__contains__("xiacheng"):
        df['area_xc']=np.array(1).astype(np.int)
    elif  dfName.__contains__("xiaoshan"):
        df['area_xs']=np.array(1).astype(np.int)
    elif  dfName.__contains__("yuhang"):
        df['area_yh']=np.array(1).astype(np.int)

    df.to_csv('Data_washed_'+dfName.split('_')[1]+dfName.split('_')[2],sep=',',index=None)

if __name__ == '__main__':
    #for i in ['binjiang','gongshu','xiacheng','xihu','yuhang','xiaoshan']:
    for i in ['binjiang']:
        for j in [1,2,3,4]:
            print(i)
            print(j)
            Wash(i,j)
