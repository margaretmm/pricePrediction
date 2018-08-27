import pandas as pd

dfName='Data_washed_all.csv'
df = pd.read_csv(dfName, sep=',')
pd.set_option('max_colwidth',100)
pd.set_option('display.max_columns', None)

#col=[Rooms,Area,Decorate,subway,FiveYear,hasLift,TotalPrice,Price,Toward_s,Toward_n,Toward_e,Toward_w,Floor_h,Floor_m,Floor_l,BuyYesrs<3,BuyYesrs_3_5,BuyYesrs_6_10,BuyYesrs>10]
df=df.fillna(0)
df.drop_duplicates()

df.to_csv('Data_washed_all2.csv',sep=',',index=None)
