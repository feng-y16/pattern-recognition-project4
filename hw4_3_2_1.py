from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
#######################################################################################################
r=np.load("train_data_eig.npz")#载入数据
train_data_eig0=r["arr_0"]
train_data_eig1=r["arr_1"]
r=np.load("10genes.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
train_data_extracted=np.dot(matrix(train_data_eig1[:,0:2]).T,matrix(train_data).T).T#特征提取和调用C means算法
test_data_extracted=np.dot(matrix(train_data_eig1[:,0:2]).T,matrix(test_data).T).T
train_label_pred = KMeans(n_clusters=2, random_state=9).fit_predict(train_data_extracted)
#######################################################################################################
train_data_extracted=train_data_extracted.getA()#求各个聚类的均值
test_data_extracted=test_data_extracted.getA()
test_label_pred=[]
label_one_num=0
label_two_num=0
label_one_sum=np.zeros(2)
label_two_sum=np.zeros(2)
for i in range(0,len(train_data_extracted)):
    if train_label_pred[i]==0:
        label_one_num=label_one_num+1
        label_one_sum[0]=label_one_sum[0]+train_data_extracted[i][0]
        label_one_sum[1]=label_one_sum[1]+train_data_extracted[i][1]
    else:
        label_two_num=label_two_num+1
        label_two_sum[0]=label_two_sum[0]+train_data_extracted[i][0]
        label_two_sum[1]=label_two_sum[1]+train_data_extracted[i][1]
mean_one=label_one_sum/label_one_num
mean_one=label_one_sum/label_one_num
mean_two=label_two_sum/label_two_num
mean_two=label_two_sum/label_two_num
#######################################################################################################
for i in range(0,len(test_data_extracted)):#最小距离分类
    distance_one_1=(test_data_extracted[i][0]-mean_one[0])*(test_data_extracted[i][0]-mean_one[0])
    distance_one_2=(test_data_extracted[i][1]-mean_one[1])*(test_data_extracted[i][1]-mean_one[1])
    distance_two_1=(test_data_extracted[i][0]-mean_two[0])*(test_data_extracted[i][0]-mean_two[0])
    distance_two_2=(test_data_extracted[i][1]-mean_two[1])*(test_data_extracted[i][1]-mean_two[1])
    if distance_one_1+distance_one_2<distance_two_1+distance_two_2:
        test_label_pred.append(0)
    else:
        test_label_pred.append(1)
#######################################################################################################
#作图
plot_Cmeans2_combined(train_data_extracted,train_label_pred,test_data_extracted,test_label_pred,mean_one,mean_two,"Normalized C means classification (2 clusters)")