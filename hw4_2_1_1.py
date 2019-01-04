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
r=np.load("train_data_eig_withoutnorm.npz")#载入数据
train_data_eig0=r["arr_0"]
train_data_eig1=r["arr_1"]
r=np.load("10genes_withoutnorm.npz")
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
data=np.dot(matrix(train_data_eig1[:,0:2]).T,matrix(train_data).T).T#特征提取
label_pred = KMeans(n_clusters=2, random_state=9).fit_predict(data)#调用C means算法
#######################################################################################################
for i in range(0,len(data)):#作图
    if label_pred[i]==0:
        plt.scatter(data[i, 0], data[i, 1],1.8, c='g',label='cluster 1')
    else:
        plt.scatter(data[i, 0], data[i, 1],1.8, c='b',label='cluster 2')
plt.xlabel('Extracted feature 1')
plt.ylabel('Extracted feature 2')
plt.title("Unnormalized C means (2 clusters)")
labelpool=['cluster 1','cluster 2']
handles, labels = plt.gca().get_legend_handles_labels() 
by_label = OrderedDict(zip(labels, handles))  
handle=[]
keys=[]
flag=[0,0]
for j in range(0,2):
    for i in range(0,len(list(by_label.keys()))):
        if list(by_label.keys())[i]==labelpool[j]:
            handle.append(list(by_label.values())[i])
            flag[j]=1
for i in range(0,2):
    if flag[i]==1:
        keys.append(labelpool[i])
plt.legend(handle, keys)
plt.savefig("C_means_withoutnorm_2clusters.png")
plt.show()