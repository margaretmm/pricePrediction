import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('Data_washed_g1.csv')  # 读取train数据
train_y = train.Price
predictor_x = ['Rooms', 'subway', 'FiveYear', 'hasLift', 'Toward_w','Floor_h']  # 特征
train_x = train[predictor_x]
my_model = RandomForestRegressor()  # 随机森林模型

my_model.fit(train_x, train_y)  # fit
test = pd.read_csv('Data_washed_g2.csv')  # 读取test数据
test_x = test[predictor_x]
test_y=test.Price
pre_test_y = my_model.predict(test_x)
print(pre_test_y)

my_submission = pd.DataFrame({'TotalPrice': test.TotalPrice, 'Price': pre_test_y,'ActualPrice': test_y,'loss':pre_test_y-test_y})  # 建csv
my_submission.to_csv('submission2.csv', index=False)