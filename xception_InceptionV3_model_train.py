# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:23:28 2017

@author: Administrator
"""

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,LearningRateScheduler
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
from keras.layers import Input
from keras.layers import Dense,Dropout,Lambda
from keras.applications import Xception,InceptionV3
from keras.models import Model
from keras import regularizers 
from keras import optimizers
import keras

    
train_datagen = ImageDataGenerator(
        rescale=1./255,   
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 16
train_generator = train_datagen.flow_from_directory(
        './train_split',
        # '/home/cwh/coding/data/cwh/test1',
        target_size=(299, 299),
        # batch_size=1,
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        './val_split',
        # '/home/cwh/coding/data/cwh/test1',
        target_size=(299, 299),
        # batch_size=1,
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical')


def double_generator(cur_generator, batch_size, train=True):
    cur_cnt = 0
    while True:
        if train and cur_cnt % 4 == 1:
            # provide same image
            x1, y1 = train_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = train_generator.next()
            # print(y1)
            # print(np.sort(np.argmax(y1, 1), 0))
            y1_labels = np.argmax(y1, 1)
            has_move = list()
            last_not_move = list()
            idx2 = [-1 for i in range(batch_size)]

            for i, label in enumerate(y1_labels):
                if i in has_move:
                    continue
                for j in range(i+1, batch_size):
                    if y1_labels[i] == y1_labels[j]:
                        idx2[i] = j
                        idx2[j] = i
                        has_move.append(i)
                        has_move.append(j)
                        break
                if idx2[i] == -1:
                    # same element not found and hasn't been moved
                    if len(last_not_move) == 0:
                        last_not_move.append(i)
                        idx2[i] = i
                    else:
                        idx2[i] = last_not_move[-1]
                        idx2[last_not_move[-1]] = i
                        del last_not_move[-1]
            x2 = list()
            y2 = list()
            for i2 in range(batch_size):
                x2.append(x1[idx2[i2]])
                y2.append(y1[idx2[i2]])
            # print(y2)
            x2 = np.asarray(x2)
            y2 = np.asarray(y2)
            # print(x2.shape)
            # print(y2.shape)
        else:
            x1, y1 = cur_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = cur_generator.next()
            x2, y2 = cur_generator.next()
            if y2.shape[0] != batch_size:
                x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        one_hot_same = np.zeros([batch_size, 2])
        one_hot_same[np.arange(batch_size), same] = 1
        # print same
        # print one_hot_same
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        yield [x1, x2], [y1, y2, one_hot_same]


def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2
'''
def lr_decay(epoch):
    lrs = [0.0001, 0.0001, 0.0001,0.0001,0.00001, 0.000001, 0.000001, 0.00001, 0.000001, 
           0.000001, 0.000001, 0.000001,
           0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
           
    return lrs[epoch]
'''    
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#my_lr = LearningRateScheduler(lr_decay)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
save_model = ModelCheckpoint('xcep_incep{epoch:01d}-{val_ctg_out_1_acc:.4f}-{val_ctg_out_2_acc:.4f}.h5', period=1)


    # create the base pre-trained model
input_tensor = Input(shape=(299, 299, 3))
base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    
base_model.layers.pop()
base_model.outputs = [base_model.layers[-1].output]
base_model.layers[-1].outbound_nodes = []
base_model.output_layers = [base_model.layers[-1]]

base_model1 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    
base_model1.layers.pop()
base_model1.outputs = [base_model1.layers[-1].output]
base_model1.layers[-1].outbound_nodes = []
base_model1.output_layers = [base_model1.layers[-1]]

feature = base_model
feature_=base_model1
img1 = Input(shape=(299, 299, 3), name='img_1')
img2 = Input(shape=(299, 299, 3), name='img_2')

feature1 = feature(img1)

feature2 = feature_(img2)

    # let's add a fully-connected layer
    #add l2 regularizer


category_predict1 = Dense(100, activation='softmax', name='ctg_out_1')(
        Dropout(0.5)(feature1)
    )
category_predict2 = Dense(100, activation='softmax', name='ctg_out_2')(
        Dropout(0.5)(feature2)
    )

    # concatenated = keras.layers.concatenate([feature1, feature2])
dis = Lambda(eucl_dist, name='square')([feature1, feature2])
    # concatenated = Dropout(0.5)(concatenated)
    # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(concatenated)

judge = Dense(2, activation='softmax', name='bin_out')(dis)
model = Model(input=[img1, img2], output=[category_predict1, category_predict2, judge])

    # model.save('dog_xception.h5')
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:

cur_base_model = model.layers[2]
for layer in cur_base_model.layers[:105]:
    layer.trainable = False
for layer in cur_base_model.layers[105:]:
    layer.trainable = True

inception_model=model.layers[3]
for layer1 in inception_model.layers[:172]:
   layer1.trainable = False
for layer1 in inception_model.layers[172:]:
   layer1.trainable = True
   

    # compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss={'ctg_out_1': 'categorical_crossentropy',
                        'ctg_out_2': 'categorical_crossentropy',
                        'bin_out': 'categorical_crossentropy'},
                  loss_weights={
                      'ctg_out_1': 1.,
                      'ctg_out_2': 1.,
                      'bin_out': 0.5
                  },
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

model.fit_generator(double_generator(train_generator, batch_size=batch_size),
                        steps_per_epoch=16806/batch_size+1,
                        #samples_per_epoch=16806//batch_size+1,                       
                        nb_epoch=50,
                        validation_data=double_generator(validation_generator, train=False, batch_size=batch_size),
                        validation_steps=1898/batch_size+1,
                        #nb_val_samples=1898//batch_size+1,                        
                        callbacks=[auto_lr,save_model])
    
model.save('dog_xception_tuned.h5')


