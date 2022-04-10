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
#sample inputs
train_ds,val_ds,test_ds=ds.import_dataset(seed=42,image_size=image_size,augment_data=True)
"""
Ensure best model already present in same directry
Download : https://drive.google.com/file/d/1bdMa03-Jf-zlZi1zL1IQLGThvNfAfHAz/view
"""
from tensorflow import keras
model = keras.models.load_model('model-best.h5')   
                         

""""
Question 5)
Guided Back propogation

"""

"""### Get all layers associated with the model"""

all_layers=[]
for l in model.layers:
  if('conv' in l.name):
    all_layers.append(l.name )

"""### Guided Backprop"""

all_act_Layer_Names=[]
for layer in model.layers:
    if('activation' in layer.name ):
        all_act_Layer_Names.append(layer.name)

all_Conv_Layer_Names=[]
for layer in model.layers:
    if('conv' in layer.name ):
        all_Conv_Layer_Names.append(layer.name)


@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad


def GuidedBackprop(model,  Nuber_of_neurons=10):
  LAYER_NAME = all_Conv_Layer_Names[-1]
  train_dir='inaturalist_12K/train/'
  test_dir='inaturalist_12K/val/'
  classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
  
  #Apply guided back prop on conv 5 layer
  guided_backprop_model = tf.keras.models.Model(
    inputs = [model.inputs],    
    outputs = [model.get_layer(LAYER_NAME).output])

  #Get all activation layers
  activations_layer = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer,'activation')]
    
  #apply guided relu
  for layer in activations_layer:
    if layer.activation == tf.keras.activations.relu:
      #apply guided Relu
      layer.activation = guidedRelu
    
  #select random image from test directory
  #random_category=random.choice(range(len(classes)))
  random_category=2
  category = classes[random_category]
  dir=os.path.join(test_dir,category)

  files=os.listdir( dir)
  
  #imgpath=random.choice(files)
  imgpath="cf9ba75f4ecf6cbdf0ba06d7fa8d9534.jpg"
 
 
  rows =12
  columns=1
  img_path=(os.path.join(dir,imgpath))
  imgs=cv2.imread(img_path)
  img=cv2.resize(imgs,(300,300))
  
  plt.imshow(imgs)
  plt.axis('off')
  plt.title('Original Image:'+str(category))
  #plt.imshow(img)
  x = np.expand_dims(img, axis=0)
  i=2
  j =3
  k=0
  while k <= Nuber_of_neurons:
    
    with tf.GradientTape() as tape:       
      inputs = tf.cast(x, tf.float32)
      tape.watch(inputs)
      outputs = guided_backprop_model(inputs)[0]
     
      out = outputs[i,j,k+20]
      grads = tape.gradient(out,inputs)[0]
      
      guided_back_prop =grads
      gb_viz = np.dstack((
                  guided_back_prop[:, :, 0],
                  guided_back_prop[:, :, 1],
                  guided_back_prop[:, :, 2],
              ))       
      gb_viz -= np.min(gb_viz)
      gb_viz /= gb_viz.max()
      plt.imshow(gb_viz)
      
      plt.axis('off')
      
      plt.title('neuron'+str(k))
      plt.show()
      #just ensure not picking same neuron
      j+=1
      i+=1
      k+=1       
     

GuidedBackprop(model,10)
