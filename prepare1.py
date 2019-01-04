import numpy as np
import pandas as pd
import sklearn
import math
from numpy import *
from matplotlib import pyplot as plt
from shuffle_fy import shuffledata
from draw import *
from sklearn import preprocessing
offset=1
#######################################################################################################
train_original_data=pd.read_csv("F:\\pythonhomework\\hw1\\E3E5_data\\train_data_E3E5_10genes.txt",sep=' ')#处理训练集
train_size=len(train_original_data["BCL2L10"])
train_data=np.zeros((train_size,10))
train_label=np.zeros(train_size)
label_num=[0,0]
label_num_total=[0,0]
for i in range(0,train_size):
    train_data[i,0]=train_original_data.ix[i]["BCL2L10"]
    train_data[i,1]=train_original_data.ix[i]["ZAR1L"]
    train_data[i,2]=train_original_data.ix[i]["C3orf56"]
    train_data[i,3]=train_original_data.ix[i]["BTG4"]
    train_data[i,4]=train_original_data.ix[i]["TUBB8"]
    train_data[i,5]=train_original_data.ix[i]["SH2D1B"]
    train_data[i,6]=train_original_data.ix[i]["C9orf116"]
    train_data[i,7]=train_original_data.ix[i]["TMEM132B"]
    train_data[i,8]=train_original_data.ix[i]["CA4"]
    train_data[i,9]=train_original_data.ix[i]["FAM19A4"]
    if train_original_data.ix[i]["label"]=="E3":
        train_label[i]=0
        label_num[0]=label_num[0]+1
        label_num_total[0]=label_num_total[0]+1
    else:
        train_label[i]=1
        label_num[1]=label_num[1]+1
        label_num_total[1]=label_num_total[1]+1
#######################################################################################################
test_original_data=pd.read_csv("F:\\pythonhomework\\hw1\\E3E5_data\\test_data_E3E5_10genes.txt",sep=' ')#处理测试集
test_size=len(test_original_data["BCL2L10"])
test_data=np.zeros((test_size,10))
test_label=np.zeros(test_size)
for i in range(0,test_size):
    test_data[i,0]=test_original_data.ix[i]["BCL2L10"]
    test_data[i,1]=test_original_data.ix[i]["ZAR1L"]
    test_data[i,2]=test_original_data.ix[i]["C3orf56"]
    test_data[i,3]=test_original_data.ix[i]["BTG4"]
    test_data[i,4]=test_original_data.ix[i]["TUBB8"]
    test_data[i,5]=test_original_data.ix[i]["SH2D1B"]
    test_data[i,6]=test_original_data.ix[i]["C9orf116"]
    test_data[i,7]=test_original_data.ix[i]["TMEM132B"]
    test_data[i,8]=test_original_data.ix[i]["CA4"]
    test_data[i,9]=test_original_data.ix[i]["FAM19A4"]
    if test_original_data.ix[i]["label"]=="E3":
        test_label[i]=0
        label_num_total[0]=label_num_total[0]+1
    else:
        test_label[i]=1
        label_num_total[1]=label_num_total[1]+1
#######################################################################################################
np.savez("10genes_original.npz",train_data,train_label,test_data,test_label,label_num,train_size,test_size,label_num_total)