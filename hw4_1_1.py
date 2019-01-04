import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
#######################################################################################################
r=np.load("10genes_withoutnorm.npz")#载入数据
train_data=r["arr_0"]
train_label=r["arr_1"]
test_data=r["arr_2"]
test_label=r["arr_3"]
label_num=r["arr_4"]
train_size=r["arr_5"]
test_size=r["arr_6"]
#######################################################################################################
train_data_cov=matrix(np.cov(train_data.T))*(len(train_data)-1)/len(train_data)#求本征值和本征向量
train_data_eig = np.linalg.eig(train_data_cov)
#######################################################################################################
eig_num=[]#作图
for i in range(0,len(train_data_eig[0])):
    eig_num.append(i+1)
eig_sum=train_data_eig[0]
ploteig(eig_num,eig_sum,'Unnormalized Eigenvalues')
np.savez("train_data_eig_withoutnorm.npz",train_data_eig[0],train_data_eig[1])
print(train_data_eig[0])
#[23.32639529  3.22140101  2.80040317  1.50452458  0.46469819  0.55059762  0.6330043   0.86801798  1.10371212  1.07011912]