# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:23:26 2017

@author: Administrator
"""

from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import re
from keras.models import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 10593
nbr_augmentation = 5

root_path_1 = './'
#狗名字的列表
file1=open(os.path.join(root_path_1,'clsori2new.txt'),'r')
lines=file1.readlines()
DogNames=[]
for line in lines[1:]:
    DogNames.append((line.split())[0])

#root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
root_path = './'
weights_path = 'xception07-0.77.h5'

test_data_dir = os.path.join(root_path, 'test/')

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)


print('Loading model and weights from training process ...')
Xecption_model = load_model(weights_path)
single_model=Model(input=Xecption_model.layers[0].input,output=Xecption_model.layers[6].output)

for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames
    #print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = single_model.predict_generator(test_generator, nbr_test_samples)
    else:
        predictions += single_model.predict_generator(test_generator, nbr_test_samples)

predictions /= nbr_augmentation

test_names=[]
for te in test_image_list:
    test_name=re.findall(r"test1(.*).jpg",te)
    test_name1=test_name[0][1:]
    test_names.append(test_name1)
   
prs=[]

for prediction in predictions:
    pr=np.where(prediction==np.max(prediction))
    
    prs.append(pr[0][0])
ww={}    
for i in range(len(test_names)):
    ww[test_names[i]]=prs[i]
root_path3 ='E:/百度狗识别比赛/clsori2new.txt'
file1=open(root_path3,'r')
ll1=[]
lines1=file1.readlines()
for line in lines1[1:]:
    line11=line.split()
    ll1.append(line11)

dd={}
for dl in ll1:
    dd[dl[0]]=dl[1]
ddw={}
for key2,value2 in ww.items():
    for key3,value3 in dd.items():
        if value2==int(value3):
            ddw[key2]=int(key3)

results=[]

for key4,value4 in ddw.items():
    result=str(value4)+'\t'+key4
    results.append(result)
    
    
final_file=open('submission.txt','w')
for rr in results:
    final_file.write(rr)
    final_file.write("\n")
final_file.close()