import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


def add_layer(inputs,input_size,output_size,activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size,output_size]),name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1,output_size]) + 0.1,name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)

dfTrain='Data_washed_all.csv'
dfTest='Data_washed_g2.csv'
keep_prob=1
df_ori = pd.read_csv(dfTrain,header=0)
df_ori_test = pd.read_csv(dfTrain,header=0)
df=df_ori[df_ori['Area']>50]# filter non normal house
df_test=df_ori[df_ori_test['Area']>50]# filter non normal house

arr=["Rooms","Area","Decorate","subway","FiveYear","hasLift","Toward_s","Toward_n","Toward_e","Toward_w","Floor_h","Floor_m","Floor_l"]
df_train=scale(df[arr])
df_train_test=scale(df_test[arr])


#df["Price"]=df["Price"].apply(lambda x:normalize(x))
df_target=scale(df["Price"].values.reshape(len(df),1))
df_target_test=scale(df_test["Price"].values.reshape(len(df_test),1))
print(np.shape(df_train))
print(np.shape(df_target))
#print("~~~~")
#print(np.random.permutation(len(df_target)))

len=len(arr)
X=tf.placeholder("float",shape=[None,len])
Y=tf.placeholder("float",shape=[None,1])
# Weight=tf.Variable(tf.random_normal([len,1]))
# Bias=tf.Variable(tf.random_normal([1]))
keep_prob_s = tf.placeholder(dtype=tf.float32)


with tf.name_scope("layer_1"):
    l1 = add_layer(X,len,1,activation_function=tf.nn.relu)
# with tf.name_scope("layer_2"):
#     l2 = add_layer(l1,6,10,activation_function=tf.nn.relu)
with tf.name_scope("y_pred"):
    pred = add_layer(l1,1,1)

# 这里多于的操作，是为了保存pred的操作，做恢复用。我只知道这个笨方法。
pred = tf.add(pred,0,name='pred')
#output=tf.matmul(X,Weight)+Bias
#pred=tf.cast(tf.sigmoid(output)>0.5,tf.float32)

with tf.name_scope("loss"):
#loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=output))
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y - pred),reduction_indices=[1]))  # mse
#train_step=tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)
#accuracy=tf.reduce_sum(tf.cast(tf.equal(pred,Y),tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
loss_train=[]
train_acc=[]
test_acc=[]

# 训练数据
for i in range(5000):
    index=np.random.permutation(np.shape(df_target)[0])
    df_train=df_train[index]
    df_target=df_target[index]
    for n in range(np.shape(df_target)[0]//100+1):
        batch_xs=df_train[n*100:n*100+100]
        batch_ys=df_target[n*100:n*100+100]
        sess.run(train_step,feed_dict={X: batch_xs,Y:batch_ys,keep_prob_s:keep_prob})
    if i%100==0:
        loss_temp=sess.run(loss,feed_dict={X:batch_xs,Y:batch_ys,keep_prob_s:keep_prob})
        loss_temp_test = sess.run(loss, feed_dict={X: df_train_test, Y: df_target_test,keep_prob_s:keep_prob})
        print("epoch:%d\tloss:%.5f\ttest loss:%.5f" % (i, loss_temp,loss_temp_test))


#
# print('b:' + str(sess.run(b)) + ' || a:' + str(sess.run(a)))





