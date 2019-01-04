import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
#######################################################################################################
r=np.load("10genes.npz")#载入数据
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
ploteig(eig_num,eig_sum,'Normalized Eigenvalues')
np.savez("train_data_eig.npz",train_data_eig[0],train_data_eig[1])
print(train_data_eig[0])
#[5.71022527 1.17465217 0.91179818 0.62460201 0.37218515 0.11907037 0.26611207 0.218721 0.15835048 0.18618367]