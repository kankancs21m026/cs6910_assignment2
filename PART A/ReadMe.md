# PART A
# Pre-requists 

**Python packages**	
Please install following packages in your local machine.
- pip install wget
- pip install opencv-python
- pip install tensorflow
- pip install keras
- pip install twilio
- pip install  pyttsx3
- pip install  argparse
- pip install h5py



**Guide to Execute Code**
# 
# Google Colab

 It is **recommanded** to run all the code in **Google Colab**.Please upload the python notebook files available under [Note Book](https://github.com/kankancs21m026/cs6910_assignment2/tree/main/PART%20A/NoteBooks) subdirectory under **PART_A** directory.For more tutorial related to Google colab usage follow the link: [Google Colab](https://colab.research.google.com/)

 | Question  | Link  |
| --- | ----------- | 
|Question 1,2,3| [NoteBook](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/NoteBooks/PartA_Question1_2_3_Sweep.ipynb)|
|Question 4,5| [NoteBook](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/NoteBooks/PartA_Qestion_4_5.ipynb)|

# Command Line

## Pre-requisite
It is optional to download and unzip iNeuralist datast before executing the programs.All program will automatically download these.
Links:
- Download the zip file from following link and place the file under parent directry  **PART A**.
[nature_12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)
- Unzip the file in same directry


Assuming you currently in **PART A** directory
Run following files sequentially

## Main package



 | File  | Link  |
| --- | ----------- | 
|CNN  | [CNN.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/utility/CNN.py)|
|Import Dataset | [Dataset.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/utility/Dataset.py)|



 | Question  | Link  |
| --- | ----------- | 
|Question 1| [PartA_Question1_CNN.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_Question1_CNN.py)|
|Question 1,2,3| [PartA_Question2_Sweep.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_Question2_Sweep.py)|
|Question 4,5| [PartA_parta_qestion_4_5.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/PartA_parta_qestion_4_5.py)|


##  Question 1
Command:

python PartA_Question1_CNN.py 

| Param  | Accepted Values | Description|Required|
| --- | ----------- | ----------- |----------- |
|filterOrganization *| ['config_all_same','config_incr','config_decr','config_alt_incr','config_alt_decr','Custom']| Filter organization |Yes|
| no_of_filters |  Comma delimited list input Example "64,64,64,64,64"  | Name of the optimizer| Only when filterOrganization selected as **custom** |
| optimizer |  [Adam,sgd]  | Name of the optimizer|No|
| lr | Any Float value ex. 0.0001 |Learning Rates|No|
| dropout | Any Float value ex. 0.0001 |dropout Rates|No|
| image_size |  integer  |size of the image |No|
| batchNormalization |  Bool  |Batch Normalisation |No|
| augment_data |  Bool  |Preprocess data |No|
| number_of_neurons_in_the_dense_layer |  integer  |size of the dense layer |No|
| activation_function| string | Activation function|No|
| wandbLog |  Bool  |Log in Wandb  |No|
* filterOrganization
- **config_all_same** : [64,64,64,64,64]
- **config_incr** : [16,32,64,128,256]
- **config_decr** : [256,128,64,32,16]
- **config_alt_incr** : [32,64,32,64,32]
- **config_alt_decr** : [64,32,64,32,64]
- **Custom** :  Any custome configuration as input.Respective parameters **no_of_filters**

Example
Please use any of the command and change respective parameters **no_of_filters** is optional parameter

**filterOrganization as "config_all_same"** Please note in this case 
```
python PartA_Question1_CNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "config_all_same" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" 
```

**filterOrganization as "custom"**
```
python PartA_Question1_CNN.py --optimizer "Adam" --lr "0.0001" --dropout "0.2" --image_size "224" --batchNormalization "True" --epoch "2" --filterOrganization "custom" --activation_function "relu" --number_of_neurons_in_the_dense_layer "256" --augment_data "True" --no_of_filters "64,64,64,64,64"
```



##  Question 2
Running sweep configuration
```
python PartA_Question2_Sweep.py
```



##  Question 4,5
Run the best model 

### Pre-requisite

In case there is no file **model-best.h5** in PARTA directry , please run the following command
```
PartA_parta_qestion_4_BestModel.py
```
**Alternatively** please download the model from below link 
https://drive.google.com/file/d/1bdMa03-Jf-zlZi1zL1IQLGThvNfAfHAz/view?usp=sharing

View the run in wanDb:
https://wandb.ai/kankan-jana/CS6910_Assignment-2/runs/179siwiu/files?workspace=user-kankan-jana

Download the model (image given below)

![image](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20A/image/wandb.jpg)

After that  run below set of commands
#### Question 4

```
python PartA_parta_qestion_4.py
```
#### Question 5

```
python PartA_parta_qestion_5.py
```
