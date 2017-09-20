# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:02:01 2017

@author: Administrator
"""
import shutil
import os
import numpy as np
#切分训练集和验证集
root_train = 'E:/百度狗识别比赛/train_split'
root_val = 'E:/百度狗识别比赛/val_split'

root_total = 'E:/百度狗识别比赛/train'

DogNames=[str(i) for i in range(100)]


root_total_path='data_train.txt'

file=open(root_total_path,'r')
lines=file.readlines()
total_root=[]
for line in lines:
    total_root.append(line.split(' '))          
for tt in total_root:
    tt[0]='.'+tt[0]          
train_path=[t[0] for t in total_root]
train_label=[t[1] for t in total_root]
tr=[]
#把训练数据集分类

for dog in DogNames:
    if dog not in os.listdir(root_total):
        os.mkdir(os.path.join(root_total, dog))

for tt in total_root:
    for dog in DogNames:
        if int(tt[1])==int(dog):
            shutil.move(tt[0],os.path.join(root_total, dog))
             
       
    
'''
nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8          
for dog in DogNames:
    if dog not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, dog))
        
    total_images = os.listdir(os.path.join(root_total, dog))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]
    
'''