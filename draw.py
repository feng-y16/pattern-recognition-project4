import numpy as np
import math
from numpy import *
from collections import OrderedDict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plotline(data,label,decide,size,feature):
    plt.scatter(x=data[:,0],y=data[:,1],s=10,c=label)
    plt.show()
    return 0
#######################################################################################################
def plotall(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None):
    colors = ['b','g','r','orange']#作分图
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    for i in range(0,train_size):
        if train_label[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=9,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    plt.plot(linex,liney,color="red")
    plt.xlabel('normalized_log(TUBB8+1)')
    plt.ylabel('normalized_log(CA4+1)')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0]
    for j in range(0,4):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,4):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys,loc = 'upper left')
    plt.title("SVM_2features")
    #plt.savefig("H2_2features.png")
    plt.show()
    return 0
#######################################################################################################
def plot3d(train_data,train_label,train_decide,test_data,test_label,test_decide,train_size,test_size,test_exist=False,linex=None,liney=None,meshz=None):
    colors = ['b','g','r','orange']#作分图(3D)
    labelpool=['E3_right','E3_wrong','E5_right','E5_wrong']
    train_decide=train_decide.astype(int)
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0,train_size):
        if train_label[i]==0:
            ax.scatter(xs=train_data[i,0],ys=train_data[i,1],zs=train_data[i,2],s=9,c=colors[train_decide[i]],marker='o',label=labelpool[train_decide[i]])
        else:
            ax.scatter(xs=train_data[i,0],ys=train_data[i,1],zs=train_data[i,2],s=9,c=colors[train_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    if(test_exist):
        test_decide=test_decide.astype(int)
        for i in range(0,test_size):
            if test_label[i]==0:
                ax.scatter(xs=test_data[i,0],ys=test_data[i,1],zs=train_data[i,2],s=9,c=colors[test_decide[i]],marker='o',label=labelpool[train_decide[i]])
            else:
                ax.scatter(xs=test_data[i,0],ys=test_data[i,1],zs=train_data[i,2],s=9,c=colors[test_decide[i]],marker='v',label=labelpool[3-train_decide[i]])
    x, y = np.meshgrid(linex, liney)
    ax.plot_surface(x, y, meshz, rstride=1, cstride=1, cmap=plt.cm.jet)
    ax.set_xlabel('normalized_log(BTG4+1)')
    ax.set_ylabel('normalized_log(SH2D1B+1)')
    ax.set_zlabel('normalized_log(CA4+1)')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0]
    for j in range(0,4):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,4):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys,loc = 'upper left')
    plt.title("SVM_3features")
    plt.show()
    return 0
#######################################################################################################
def ploteig(eig_num,eig_sum,title):
    plt.plot(eig_num,eig_sum,color="red")
    plt.xlabel('i')
    plt.ylabel('lambda_i')
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()
    return 0
#######################################################################################################
def plot_Cmeans2_combined(train_data,train_predict,test_data,test_predict,mean1,mean2,title):
    colors = ['g','b','orange','purple']
    labelpool=['cluster 1 (train data)','cluster 1 (test data)','cluster 1 mean','cluster 2 (train data)','cluster 2 (test data)','cluster 2 mean']
    for i in range(0,len(train_data)):
        if train_predict[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=2,c=colors[train_predict[i]],marker='o',label=labelpool[0])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=2,c=colors[train_predict[i]],marker='o',label=labelpool[3])
    for i in range(0,len(test_data)):
            if test_predict[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=2,c=colors[test_predict[i]+2],marker='v',label=labelpool[1])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=2,c=colors[test_predict[i]+2],marker='v',label=labelpool[4])
    plt.scatter(mean1[0], mean1[1],100, c='g',marker='x',label='cluster 1 mean')
    plt.scatter(mean2[0], mean2[1],100, c='b',marker='x',label='cluster 2 mean')
    plt.xlabel('Extracted feature 1')
    plt.ylabel('Extracted feature 2')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0,0,0]
    for j in range(0,6):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,6):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys)
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()
    return 0
#######################################################################################################
def plot_Cmeans3_combined(train_data,train_predict,test_data,test_predict,mean1,mean2,mean3,title):
    colors = ['g','b','pink','black','purple','orange']
    labelpool=['cluster 1 (train data)','cluster 1 (test data)','cluster 1 mean','cluster 2 (train data)','cluster 2 (test data)','cluster 2 mean','cluster 3 (train data)','cluster 3 (test data)','cluster 3 mean']
    for i in range(0,len(train_data)):
        if train_predict[i]==0:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=3,c=colors[train_predict[i]],marker='o',label=labelpool[0])
        elif train_predict[i]==1:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=3,c=colors[train_predict[i]],marker='o',label=labelpool[3])
        else:
            plt.scatter(x=train_data[i,0],y=train_data[i,1],s=3,c=colors[train_predict[i]],marker='o',label=labelpool[6])
    for i in range(0,len(test_data)):
            if test_predict[i]==0:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=3,c=colors[test_predict[i]+3],marker='v',label=labelpool[1])
            elif test_predict[i]==1:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=3,c=colors[test_predict[i]+3],marker='v',label=labelpool[4])
            else:
                plt.scatter(x=test_data[i,0],y=test_data[i,1],s=3,c=colors[test_predict[i]+3],marker='v',label=labelpool[7])
    plt.scatter(mean1[0], mean1[1],100, c=colors[0],marker='x',label='cluster 1 mean')
    plt.scatter(mean2[0], mean2[1],100, c=colors[1],marker='x',label='cluster 2 mean')
    plt.scatter(mean3[0], mean3[1],100, c=colors[2],marker='x',label='cluster 3 mean')
    plt.xlabel('Extracted feature 1')
    plt.ylabel('Extracted feature 2')
    handles, labels = plt.gca().get_legend_handles_labels()  
    by_label = OrderedDict(zip(labels, handles))  
    handle=[]
    keys=[]
    flag=[0,0,0,0,0,0,0,0,0]
    for j in range(0,9):
        for i in range(0,len(list(by_label.keys()))):
            if list(by_label.keys())[i]==labelpool[j]:
                handle.append(list(by_label.values())[i])
                flag[j]=1
    for i in range(0,9):
        if flag[i]==1:
            keys.append(labelpool[i])
    plt.legend(handle, keys)
    plt.title(title)
    plt.savefig(title+".png")
    plt.show()
    return 0