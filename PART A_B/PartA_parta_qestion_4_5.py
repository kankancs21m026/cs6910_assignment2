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
#Parameters
number_of_neurons_in_the_dense_layer=512
augment_data=True
filterOrganization='config_incr'
activation_function='relu'
dropout=0.2
optimizer='adam'
lr=0.0001
BatchNormalization=True
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

epoch=25
wandbLog=False
model=cnn.train(model,train_ds,val_ds,optimizer,lr,epoch,wandbLog)

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



""""
Question 5)
Guided Back propogation

"""

"""### All layers associated with the model"""

all_layers=[]
for l in model.layers:
  if('conv' in l.name):
    all_layers.append(l.name )

"""### Feature Map"""

filter_cnt=1
#output of first layer
outputs = [model.get_layer(all_layers[1]).output]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = outputs)
x   = tf.keras.utils.img_to_array(image)                           
x   = x.reshape((1,) + x.shape)
x /= 255.0

feature_map = visualization_model.predict(x)

 
n_features = feature_map.shape[-1]  # number of features in the feature map
size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

"""# Postprocess the feature 
for i in range(n_features):
  x  = feature_map[0, :, :, i]
  x -= x.mean()
  x /= x.std ()
  x *=  64
  x += 128
  x  = np.clip(x, 0, 255).astype('uint8')
  feature_map[0, :, :, i]=x"""


fig = plt.figure(figsize=(10,10))

#plotting all the filters
for i in range(n_features):  
    fig.add_subplot(int(4),int(n_features/4),i+1)
    plt.imshow(feature_map[0, :, :, i], aspect='auto')
    filter_cnt+=1
    plt.axis('off')
plt.show()

"""### Guided Backprop"""

all_act_Layer_Names=[]
for layer in model.layers:
    if('activation' in layer.name ):
        all_act_Layer_Names.append(layer.name)
print(all_act_Layer_Names)

all_Conv_Layer_Names=[]
for layer in model.layers:
    if('conv' in layer.name ):
        all_Conv_Layer_Names.append(layer.name)
print(all_Conv_Layer_Names)

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
  layer_activations = [layer for layer in guided_backprop_model.layers[1:] if hasattr(layer,'activation')]
    
  #apply guided relu
  for layer in layer_activations:
    if layer.activation == tf.keras.activations.relu:
      layer.activation = guidedRelu
    
  #select random image from test directory
  #random_cat=random.choice(range(len(classes)))
  random_cat=8
  category = classes[random_cat]
  dir=os.path.join(test_dir,category)

  files=os.listdir( dir)
  
  #imgpath=random.choice(files)
  imgpath="b4da1ba61661a95f12e8f9c653330222.jpg"
 
 
  rows =12
  columns=1
  img_path=(os.path.join(dir,imgpath))
  imgs=cv2.imread(img_path)
  img=cv2.resize(imgs,(300,300))
  fig = plt.figure(figsize=(50,50))
  fig.add_subplot(rows,columns,1)
  
  plt.imshow(imgs)
  plt.axis('off')
  plt.title('Original Image:'+str(category))
  #plt.imshow(img)
  x = np.expand_dims(img, axis=0)
  i=0
  j = 0
  k=1
  while k <= Nuber_of_neurons:
      
    with tf.GradientTape() as tape:       
      inputs = tf.cast(x, tf.float32)
      tape.watch(inputs)
      outputs = guided_backprop_model(inputs)[0]
      out = outputs[i,j,k]
      grads = tape.gradient(out,inputs)[0]
      
      guided_back_prop =grads
      gb_viz = np.dstack((
                  guided_back_prop[:, :, 0],
                  guided_back_prop[:, :, 1],
                  guided_back_prop[:, :, 2],
              ))       
      gb_viz -= np.min(gb_viz)
      gb_viz /= gb_viz.max()
      fig.add_subplot(rows,columns,k+2)
      plt.imshow(gb_viz)
      plt.axis('off')
      plt.title('neuron'+str(k))
      
      k+=1
      #just ensure not picking same neuron
      j+=1
      i+=1

GuidedBackprop(model,10)