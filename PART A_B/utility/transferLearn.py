"""
Following functions used to construct an CNN network

Parameters
---------
base_model: Name of the Base model
numberOfDenseLayer: Number of dense layer applied after base model
BatchNormalization: If true apply batch normalisation after each layer
number_of_neurons_dense_layer : Dense layer neuron 
number_of_classes: Total number of classes
dropout: dropout rate
BatchNormalization: Whether batch normalization applied

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
class TransferLearn:

    def train(model,train_ds,val_ds,epoch=10):
        model.compile(optimizer=tf.keras.optimizers.Adam(),\
              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],\
              metrics=['accuracy']
              )
        #adding early condition
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)
        hist=model.fit(train_ds, epochs=epoch,validation_data=val_ds,callbacks=early_stop)
        return model
    def setUp(base_model="inceptionv3",
            img_size=224,\
            activation_function="relu",\
            number_of_neurons_dense_layer=512,\
            number_of_classes=10,\
            dropout_rate=0,BatchNormalization=True,\
            layers_trainable=0,numberOfDenseLayer=1):
       
        imgshape=(img_size,img_size,3)
        if(base_model.lower()=="inceptionv3"):
            base_model_def = tf.keras.applications.InceptionV3(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=10)
            
        if(base_model.lower()=='inceptionresnetv2'):
            base_model_def = tf.keras.applications.InceptionResNetV2(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=10)
        if(base_model.lower()=='resnet50'):
            base_model_def = tf.keras.applications.ResNet152V2(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=10)
        
        if(base_model.lower()=='xception'):
            
            base_model_def = tf.keras.applications.Xception(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=10)
        
        
        #last few layers of base model will be trainable 
        #based on the parameter base_model_def
        base_model_def.tranable=False
        total_layers=len(base_model_def.layers)
        for i in range(total_layers-layers_trainable ,total_layers):
            base_model_def.layers[i].trainable=True
        
        model= Sequential()
        model.add(base_model_def) 
        model.add(Flatten()) 
        #Adding the Dense layers along with activation and batch normalization
        for i in range(numberOfDenseLayer):
            model.add(Dense(number_of_neurons_dense_layer,activation=(activation_function)))
            if (BatchNormalization):
              model.add(tf.keras.layers.BatchNormalization())
       
        if(dropout_rate>0):
            model.add(Dropout(dropout_rate))
        model.add(Dense(10,activation=('softmax'))) 
        return model

