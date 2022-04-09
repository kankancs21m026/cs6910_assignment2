"""
Following functions used to construct an CNN network

Parameters
---------
filters: number of filters in each layer.This is mendatory parameter
size_of_filters: Size of filters in each layers
BatchNormalization: If true apply batch normalisation after each layer
number_of_neurons_dense_layer : Dense layer neuron 
number_of_classes: Total number of classes
dropout: dropout rate
BatchNormalization: Whether batch normalization applied

filterSize: 
size of filter in first layer .mostly used only when filterOrganization selected

filterOrganization:
Some default configuaration selected.Optionally user can pass there own custom filter configuration from
"filters" parameter.This is optional parameter.Values are [all_same ,incr ,decr ,alt_incr ,alt_decr ]
               
config_all_same:[64,64,64,64,64]
config_incr : [16,32,64,128,256]
config_decr: [256,128,64,32,16]
config_alt_incr: [32,64,32,64,32]  
config_alt_decr:[64,32,64,32,64]              
 
"""
import numpy as np
import pandas as pd
import os
import keras
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from keras.layers import Conv2D , MaxPool2D,MaxPooling3D , Flatten , Dropout, Dense, Activation, BatchNormalization
class CNN:

    def train(model,train_ds,val_ds,optimizer="adam",lr=0.0001,epoch=5,wandbLog=False):
        if(optimizer=="sgd"):
          model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9),\
              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],\
              metrics=['accuracy']
              ) 
        else:
          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),\
              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],\
              metrics=['accuracy']
              ) 
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
        if(wandbLog):
          hist=model.fit(train_ds, epochs=epoch,validation_data=val_ds,callbacks=[early_stop,WandbCallback()])
        else:
          hist=model.fit(train_ds, epochs=epoch,validation_data=val_ds,callbacks=[early_stop])
     
        return model
    def setUp(filters=[16,32,64,128,256],\
            size_of_filters= [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],\
            activation_function="relu",\
            number_of_neurons_dense_layer=256,\
            number_of_classes=10,\
            dropout=0,\
            BatchNormalization=True,
            filterSize=32,
            filterOrganization='custom',
            imsize=300
           ):
     
        if(filterOrganization=='config_all_same'):
            filters=[64,64,64,64,64]
        if(filterOrganization=='config_incr'):
            filters=[16,32,64,128,256]
        if(filterOrganization=='config_decr'):
            filters=[256,128,64,32,16]
        if(filterOrganization=='config_alt_incr'):
            filters=[32,64,32,64,32]     
        if(filterOrganization=='config_alt_decr'):
            filters=[64,32,64,32,64]

        model = Sequential()
        num_of_filters=len(filters)
        model.add(Conv2D(filters[0], size_of_filters[0],input_shape=(imsize,imsize,3)))
        if BatchNormalization:
                model.add(tf.keras.layers.BatchNormalization()) 
        model.add(MaxPooling2D((2,2)))

        for i in range(num_of_filters-1):
            model.add(Conv2D(filters[i+1], size_of_filters[i+1]))

            model.add(Activation(activation_function))
            if BatchNormalization:
                model.add(tf.keras.layers.BatchNormalization()) 
            model.add(MaxPooling2D((2,2)))


        model.add(Flatten())

        model.add(Dense(number_of_neurons_dense_layer,activation=activation_function)) 
        if BatchNormalization:
                model.add(tf.keras.layers.BatchNormalization())
        if(dropout>0):
            model.add(Dropout(dropout))
        model.add(Dense(number_of_classes, activation='softmax'))
        return model

