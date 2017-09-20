# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:11:48 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import re

'''
test_names=[]
for te in test_image_list:
    test_name=re.findall(r"test_out(.*).jpg",te)
    test_name1=test_name[0][1:]
    test_names.append(test_name1)
   
prs=[]

for prediction in predictions:
    pr=np.where(prediction==np.max(prediction))
    
    prs.append(pr[0][0])
ww={}    
for i in range(len(test_names)):
    ww[test_names[i]]=prs[i]

root_path2 = 'E:/百度狗识别比赛/test-1 图片对应表.txt'
file=open(root_path2,'r')
ll=[]
lines=file.readlines()
for line in lines:
    line1=line.split()
    ll.append(line1)
www={}    
for l in ll:
    www[l[0]]=l[1]


wwww={}
for key,value in ww.items():
    for key1,value1 in www.items():
        if key==key1:
            wwww[value1]=value

'''
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