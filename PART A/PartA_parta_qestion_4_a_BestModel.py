from  utility.Dataset import Dataset as ds
from  utility.CNN import CNN as cnn
from wandb.keras import WandbCallback

from os.path import exists
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random 
import cv2
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
import os

cwd = os.getcwd()
train_dir='inaturalist_12K/train/'
test_dir='inaturalist_12K/val/'
classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

#Load data
ds.downloadDataSet()
image_size=300

"""
Question 4 
a) Use the best model from your sweep and report the accuracy on the test set.

"""
# Best model Parameters
number_of_neurons_in_the_dense_layer=512
augment_data=True
filterOrganization='config_incr'
activation_function='relu'
dropout=0.2
optimizer='adam'
lr=0.0001
BatchNormalization=True
epoch=25
#sample inputs
train_ds,val_ds,test_ds=ds.import_dataset(seed=42,image_size=image_size,augment_data=augment_data)
no_of_filters = [256,256,256,256,256] #optional as we already selected filterOrganization='config_incr'
size_of_filters = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

number_of_classes=10

#Train the model
model=cnn.setUp(no_of_filters,size_of_filters,\
          activation_function,\
          number_of_neurons_in_the_dense_layer,\
          number_of_classes,\
          dropout,BatchNormalization,\
          filterSize=16,\
         filterOrganization=filterOrganization,imsize=image_size) 



wandbLog=False
model=cnn.train(model,train_ds,val_ds,optimizer,lr,epoch,wandbLog)

model.evaluate(test_ds)
model.save('model-best.h5')


