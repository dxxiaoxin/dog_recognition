# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:24:56 2017

@author: Administrator
"""

import tensorflow as tf
import os
from PIL import Image 


train_data_path='./train_split'
val_data_path='./val_split'
root_path_1 = 'E:\百度狗识别比赛\百度狗识别比赛正式版\训练数据'

rows=224
cols=224
depth=3
#狗名字的列表
file1=open(os.path.join(root_path_1,'clsori2new.txt'),'r')
lines=file1.readlines()
DogNames=[]
for line in lines[1:]:
    DogNames.append((line.split())[0])

    
    
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

writer=tf.python_io.TFRecordWriter('trainData.tfrecords')
vWriter=tf.python_io.TFRecordWriter('valData.tfrecords')
for dog in DogNames:
    trainImages=os.listdir(os.path.join(train_data_path,dog))
    valImages=os.listdir(os.path.join(val_data_path,dog))
    for trainImage in trainImages:
        trImgPath=os.path.join(train_data_path,dog,trainImage)
        img=Image.open(trImgPath)
        img=img.resize((224,224))
        img_raw=img.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature = {
                                    'label':_int64_feature(int(dog)),
                                    'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    for valImage in valImages:
        valImgPath=os.path.join(val_data_path,dog,valImage)
        img=Image.open(valImgPath)
        img=img.resize((224,224))
        img_raw=img.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature = {
                                    'label':_int64_feature(int(dog)),
                                    'image_raw': _bytes_feature(img_raw)}))
        vWriter.write(example.SerializeToString())
    vWriter.close()
        
        
        
        
        
        
        
        
        