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

#sample inputs
augment_data=True
image_size=300
train_ds,val_ds,test_ds=ds.import_dataset(seed=42,image_size=image_size,augment_data=augment_data)

"""
Ensure best model already present in same directry
Download : https://drive.google.com/file/d/1bdMa03-Jf-zlZi1zL1IQLGThvNfAfHAz/view
"""
from tensorflow import keras
model = keras.models.load_model('model-best.h5')   



"""
Question 4 a)
"""
#verify it's working on test data                                   
model.evaluate(test_ds)
    


"""
Question 4 
b) Provide a 10 x 3 grid containing sample images from the test data and predictions made by your best model (more marks for presenting this grid creatively).

"""
randomSamples=test_ds.subset

classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
imagePath='inaturalist_12K/val/'
listImages=[]
listLabels=[]
columns = 5
rows = 5
imcount=1
clsLabel=0
for i in classes:
    p=imagePath+'/'+i
    files=os.listdir( p)
    img1=random.choice(files)
    imgplt1=mpimg.imread( p+'/'+str(img1))
    imgplt1=cv2.resize(imgplt1,(300,300)) 
    listImages.append(imgplt1)
    img2=random.choice(files)
    imgplt2=mpimg.imread( p+'/'+str(img2))
    imgplt2=cv2.resize(imgplt2,(300,300)) 
    listImages.append(imgplt2)
    img3=random.choice(files)
    imgplt3=mpimg.imread( p+'/'+str(img3))
    imgplt3=cv2.resize(imgplt3,(300,300)) 
    listImages.append(imgplt3) 
    listLabels.append(clsLabel)
    listLabels.append(clsLabel)
    listLabels.append(clsLabel)
    clsLabel+=1

"""### Plot 30 random images from Test Dataset and show there actual vs predicted Result"""

predictions=[]
for image in listImages:
  pred=model.predict(image.reshape(1,300,300,3)).argmax()
  predictions.append(pred)

fig = plt.figure(figsize=(15,30))

for i in range(30):
  img=cv2.resize(listImages[i],(300,300))
  fig.add_subplot(10,3,i+1)
  plt.imshow(img)
  plt.axis('off')
  plt.title('Actual: '+str(classes[listLabels[i]])+', Predicted: '+str(classes[predictions[i]]),fontdict={'fontsize':10})




"""
Question 4
C)Visualise all the filters in the first layer of your best model
 for a random image from the test set. If there are 64 filters in the first layer plot them in an 8 x 8 grid.

"""
"""### Filter visualisation RGB channels"""

#Iterate thru all the layers of the model
layer= model.layers[0]
weights, bias= layer.get_weights()
filters, biases = layer.get_weights()
#print(layer.name, filters.shape)

#normalize filter values between  0 and 1 for visualization
f_min, f_max = weights.min(), weights.max()
filters = (weights - f_min) / (f_max - f_min)

"""### Visualisation through separate R,G,B channel"""

filter_cnt=1
fig = plt.figure(figsize=(15,30))
#plotting all the filters
for i in range(filters.shape[3]):
    #get the filters
    filt=filters[:,:,:, i]
    #plotting each of the channel, color image RGB channels
    for j in range(filters.shape[0]):
        ax= plt.subplot(filters.shape[3], filters.shape[0], filter_cnt  )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(filt[:,:, j])
        filter_cnt+=1
plt.show()

"""### Filter visualisation RGB Channel combined"""

#The following plot shows the 16 filters in a 4x4 grid, with RGB channels combined.
filter_cnt=1
fig = plt.figure(figsize=(10,10))

#plotting all the filters
for i in range(filters.shape[3]):
    #get the filters
    filt=filters[:,:,:, i]
    fig.add_subplot(4,4,i+1)
   
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(filt[:,:])
    filter_cnt+=1
    plt.axis('off')
plt.show()

"""### Select an image for showing filter visualisation"""

classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
imagePath='inaturalist_12K/val'
i=random.choice(range(len(classes)))
path=imagePath+'/'+classes[i]
files=os.listdir( path)
imgchoice=random.choice(files)
image=tf.keras.preprocessing.image.load_img(path+'/'+str(imgchoice), color_mode="rgb", target_size=(300, 300))
plt.imshow(image)
plt.axis('off')
