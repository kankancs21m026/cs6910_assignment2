
from wandb.keras import WandbCallback

import os
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
class Dataset:
    #download nature_12K zip file
    def downloadDataSet(self):
      cwd = os.getcwd()
     
      classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
      file_exists = exists('./nature_12K.zip')
      if(file_exists==False):
        print('downloading....')
        os.system('curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > nature_12K.zip')
        print('download Complete')
      extract_exists = exists('./inaturalist_12K/')   
      if(extract_exists==False):  
        savePath=cwd
        savefile='./nature_12K.zip'
        print('Extracting..')
        with zipfile.ZipFile(savefile, 'r') as zip_ref:
            zip_ref.extractall(savePath)
        print('Complete')
       
        
    def showRandomImageOfEveryClass(self,imagesize=300):
       classes=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']
       imagePath='./inaturalist_12K/train'
       listImages=[]
       columns = 5
       rows = 5
       imcount=1
       
       for i in classes:
           p=imagePath+'/'+i
           files=os.listdir( p)
           img=random.choice(files)
           imgplt=mpimg.imread( p+'/'+str(img))
           imgplt=cv2.resize(imgplt,(300,300)) 
           listImages.append(imgplt)
           #plt.imshow(imgplt)
           #fig.add_subplot(rows, columns, imcount)
           #imcount+=1
           #plt.axis('off')
           #plt.title(i) 
       
       _, axs = plt.subplots(1, 10, figsize=(15, 15))
       axs = axs.flatten()
       for img, ax,cls in zip(listImages, axs,classes):
         
           ax.imshow(img)
           ax.axis('off')
           ax.set_title(str(cls))
       plt.show()
    
    #inport data set from the file
    def import_dataset(self,seed,image_size=300,augment_data=False):
        #All variables 
        cwd=os.getcwd()
        batchsize=32
        image_size=image_size
        train_dir = './inaturalist_12K/train/'
        test_dir = './inaturalist_12K/val/'
        print('Training Dataset')

        #perform Augmentation if augment_data=True
        if augment_data:
            train_datagen = ImageDataGenerator(rescale=1./255,
                                          rotation_range=90,
                                          zoom_range=0.2,
                                          shear_range=0.2,
                                          validation_split=0.1,
                                          horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1./255)
        else:
            train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.1)
            test_datagen = ImageDataGenerator(rescale=1./255)

        train_ds = train_datagen.flow_from_directory(train_dir,subset="training", class_mode='sparse',color_mode='rgb',target_size=(image_size, image_size), batch_size=batchsize)
        val_ds = train_datagen.flow_from_directory(train_dir,subset="validation",class_mode='sparse', color_mode='rgb',target_size=(image_size, image_size), batch_size=batchsize)
        print('')
        print('Test Dataset')
        test_ds = test_datagen.flow_from_directory(test_dir, target_size=(image_size, image_size) ,class_mode='sparse',color_mode='rgb',batch_size=30)
    



    
        return train_ds,val_ds, test_ds
               


# ### Transfer Learning

# In[20]:


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
batchNormalization: Whether batch normalization applied
no_of_last_trainable_layers: Last k layers remain trainable
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

    def Train(self,base_model="inceptionv3",
            img_size=224,\
            activation_function="relu",\
            number_of_neurons_dense_layer=512,\
            number_of_classes=10,\
            dropout_rate=0,batchNormalization=False,\
            fraction_of_trainable_layers=0,numberOfDenseLayer=1,lr=1e-4,optimizer="adam",wandbLog=False):
       
        imgshape=(img_size,img_size,3)
        #choose model based on inputs
        if(base_model.lower()=="inceptionv3"):
            base_model_def = tf.keras.applications.InceptionV3(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=1000)
            
        if(base_model.lower()=='inceptionresnetv2'):
           
            base_model_def = tf.keras.applications.InceptionResNetV2(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=1000)
        if(base_model.lower()=='resnet50'):
           
            base_model_def = tf.keras.applications.ResNet50(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=1000)
        
        if(base_model.lower()=='xception'):

            base_model_def = tf.keras.applications.Xception(input_shape=imgshape,
                                               include_top=False,
                                               weights='imagenet',classes=1000)
        
        #--------------------------
        #Pretraining
        #-------------------------
        base_model_def.tranable=False

        model= Sequential()
        model.add(base_model_def) 
        model.add(Flatten()) 
        #Adding the Dense layers along with activation and batch normalization
        for i in range(numberOfDenseLayer):
            model.add(Dense(number_of_neurons_dense_layer,activation=(activation_function)))
            if (batchNormalization):
              model.add(tf.keras.layers.BatchNormalization())
        
        if(dropout_rate>0):
            model.add(Dropout(dropout_rate))
        model.add(Dense(10,activation=('softmax'))) 


        if(optimizer=="sgd"):
          model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9),              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],              metrics=['accuracy']
              ) 
        else:
          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],              metrics=['accuracy']
              ) 
        #early stopping as it may lead overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,restore_best_weights=True)
        #adding early condition
        if(wandbLog):
          hist=model.fit(train_ds, epochs=5,validation_data=val_ds,callbacks=[early_stop,WandbCallback()])
        else:
          hist=model.fit(train_ds, epochs=5,validation_data=val_ds,callbacks=[early_stop])
        #------------------------
        #fine tunning
        #-------------------------
        
        #no_of_last_trainable_layers if value zero we will train all
        total_layers=len(base_model_def.layers)
        no_of_last_trainable_layers=int(total_layers*fraction_of_trainable_layers)
        for i in range(total_layers-no_of_last_trainable_layers ,total_layers):
            base_model_def.layers[i].trainable=True
  
        #now train again with very low learning rate
        if(optimizer=="sgd"):
          model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9),              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],              metrics=['accuracy']
              ) 
        else:
          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),              loss=[tf.keras.losses.SparseCategoricalCrossentropy()],              metrics=['accuracy']
              ) 

     
          
        if(wandbLog):
          hist=model.fit(train_ds, epochs=5,validation_data=val_ds,callbacks=[early_stop,WandbCallback()])
        else:
          hist=model.fit(train_ds, epochs=5,validation_data=val_ds,callbacks=[early_stop])


        return model


ds=Dataset()

# ### Download data set
ds.downloadDataSet()

# ### Load data set

train_ds,val_ds,test_ds=ds.import_dataset(seed=42,image_size=300,augment_data=True)



# Fine-tune the model using wandb runs
def train_wandb():
    run = wandb.init()
    image_shape=(300, 300)
    config=wandb.config
    # Set the run name
    name = str(config["base_model"]) + "_"
    name += " optimizer(" + str(config["optimizer"]) + ")_"
    name += " fraction_of_trainable_layers(" + str(config["fraction_of_trainable_layers"]) + ")_"
    name += "dropout_rate(" + str(config["dropout_rate"])+ ")_"
    name += "activation_function(" + str(config["activation_function"])+ ")_"
    name += "numberOfDenseLayer(" + str(config["numberOfDenseLayer"]) + ")_"
    name += "batchNormalization(" + str(config["batchNormalization"]) + ")"
    name += "lr(" + str(config["lr"]) + ")"
    wandb.run.name = name[:-1]
    #sample inputs
    cnn=TransferLearn()
    base_model=config["base_model"]
    activation_function = config["activation_function"]
    number_of_neurons_in_the_dense_layer = 512
    numberOfDenseLayer=config["numberOfDenseLayer"]
    dropout=0.3
    number_of_classes=10
    numberOfDenseLayer=1
    img_size=300
    lr=config["lr"]
    batchNormalization=config["batchNormalization"]
    optimizer=config["optimizer"]
    fraction_of_trainable_layers=config["fraction_of_trainable_layers"]
    model=cnn.Train(base_model,
                    img_size,\
                    activation_function,\
                    number_of_neurons_in_the_dense_layer,\
                    number_of_classes,\
                    dropout,\
                    BatchNormalization,\
                    fraction_of_trainable_layers,numberOfDenseLayer,lr,optimizer,wandbLog=True) 
 
import wandb
wandb.login()

#sweep configurations
sweep_config = {
  "name": "DLAssignmentQuesB",
  "method": "bayes",
  "metric": {
      "name": "validation_loss",
      "goal": "minimize",
  },
  
  "parameters": {
        "base_model": {
            "values": ["inceptionv3","resnet50", "inceptionresnetv2",  "xception"]
        },
        "lr":{
          "values":[1e-5,1e-4]  
        },
        "optimizer":{
          "values":['adam','sgd']  
        },
        "fraction_of_trainable_layers":{ "values": [1,.50,.75]},
        "dropout_rate":{ "values": [.3,0.4,0.5]},
        "batchNormalization":{ "values": [True,False]},
        "activation_function": {"values": ["relu","LeakyReLU"]},
        "numberOfDenseLayer":{"values":[1,2]}
        }
    }


sweep_id = wandb.sweep(sweep_config, entity="kankan-jana", project="CS6910_Assignment-2")


#wandb.agent("gb27evo2", train_wandb)
wandb.agent(sweep_id, train_wandb)

