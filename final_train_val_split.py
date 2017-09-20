# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:10:31 2017

@author: Administrator
"""
import numpy as np
import os
import shutil
root_path = 'E:\百度狗识别比赛\百度狗识别比赛正式版\训练数据'

root_train = './train_split'
root_val = './val_split'

root_total = './train'

#狗名字的列表
file1=open(os.path.join(root_path,'clsori2new.txt'),'r')
lines=file1.readlines()
DogNames=[]
for line in lines[1:]:
    DogNames.append((line.split())[0])
    
    
nbr_train_samples = 0
nbr_val_samples = 0

for dog in DogNames:
    if dog not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, dog))
    if dog not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val,dog))
# Training proportion
split_proportion = 0.8

for dog in DogNames:
    total_images=os.listdir(os.path.join(root_total,dog))
    num_train=int(len(total_images)*split_proportion)
    np.random.shuffle(total_images)
    train_images=total_images[:num_train]
    val_images=total_images[num_train:]
    for img in train_images:
        source=os.path.join(root_total,dog,img)
        target=os.path.join(root_train,dog,img)
        shutil.copy(source,target)
        nbr_train_samples+=1
    for img in val_images:
        source=os.path.join(root_total,dog,img)
        target=os.path.join(root_val,dog,img)
        shutil.copy(source,target)
        nbr_val_samples+=1
        
print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))