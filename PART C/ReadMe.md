# PART C
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

## Download yolov3.weights
Download file **yolov3.weights** and place it into **\PART C\model**

https://pjreddie.com/media/files/yolov3.weights

or run following command
```
wget https://pjreddie.com/media/files/yolov3.weights
```

## Changes :

Before running code it is necessary to make changes to following file
[alertSMS.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20C/alertSMS.py)
![change](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20C/image/image.jpg)

**Please login** to http://twil.io and create new account to get the details.In main file it has been masked due to **security** reasons

**Alternatively,** following file can be executed from command line , but it will not sent alert message.
[surveillanceWithoutAlert.py](https://github.com/kankancs21m026/cs6910_assignment2/blob/main/PART%20C/surveillanceWithoutAlert.py)



# Command Line

## surveillance System 

```
python surveillance.py <path/to/video/file>
```

or  run following in case no account in  http://twil.io
```
python surveillanceWithoutAlert.py <path/to/video/file>
```

## blind Assistance System 

```
python blindassistance.py <path/to/image/file>
```
