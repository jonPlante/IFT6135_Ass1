# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:47:17 2019

@author: Jonathan
"""

import os
import numpy as np
from scipy import misc
import pickle

test_set=[]
train_set=[]
valid_set=[]
splitPerc=0.8
test_path=r"./testset/test/"
testfiles = os.listdir(test_path)
print('Loading test files\n')
for file in testfiles:
    file_id=float(file[:-4])
    image=misc.imread(test_path+file, mode='RGB')
    image=np.moveaxis(image,2,0)
    image=image/255
    test_set.append([image,file_id])

print('Loading train files - cat\n')
cat_path=r"./trainset/Cat/"
catfiles = os.listdir(cat_path)
total=len(catfiles)
label=0
for i in range(total):
    image=misc.imread(cat_path+catfiles[i], mode='RGB')
    image=np.moveaxis(image,2,0)
    image=image/255
    if i/total<=splitPerc:
        train_set.append([image,label])
    else:
        valid_set.append([image,label])

print('Loading train files - dog\n')
dog_path=r"./trainset/Dog/"
dogfiles = os.listdir(dog_path)
total=len(dogfiles)
label=1
for i in range(total):
    image=misc.imread(dog_path+dogfiles[i], mode='RGB')
    image=np.moveaxis(image,2,0)
    image=image/255
    if i/total<=splitPerc:
        train_set.append([image,label])
    else:
        valid_set.append([image,label])
        
print('dumping in pickle\n')
with open('catDogData.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([train_set, valid_set, test_set], f)

print('Data ready\n')