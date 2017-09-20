# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:46:41 2017

@author: Administrator
"""
import tensorflow as tf

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    img_batch,label_batch=tf.train.shuffle_batch([img,label],batch_size=32,capacity=
                                                 2000,min_after_dequeue=1000)
    return img_batch,label_batch
    
    
    
if __name__=="__main__":
    init=tf.global_variables_initializer()
    img_batch,label_batch=read_and_decode('trainData.tfrecords')
    
        