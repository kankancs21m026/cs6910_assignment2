from ast import arg
import numpy as np
import argparse
import cv2
import os
from os.path import exists
import time
from textToSpeech import play
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from  utility import extract_boxes_confidences_classids,draw_bounding_boxes,make_prediction

import sys
import wget
#setup
cwd = os.getcwd()
file_exists = exists(cwd+'/model/yolov3.weights')
if(file_exists==False):
    url='https://pjreddie.com/media/files/yolov3.weights'
    print('Downloading..')
    wget.download(url,cwd+'/model/')
    

#args,labels=readArgs()
labels = open('model/coco.names').read().strip().split('\n')

# Create a list of colors for the labels
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
# Load weights using OpenCV
net = cv2.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')

# Get the ouput layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

#layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
file = sys.argv[1]
confidence=0.5
threshold=0.3
if('.jpg' in file.lower() or '.jpeg' in file.lower()):
    image = cv2.imread(file)

    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)

    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors,labels)
    # show the output image
    
    cv2.imshow('YOLO Object Detection', image)
    cv2.waitKey(0)

    
    cv2.imwrite(f'output/{file.split("/")[-1]}', image)
    cv2.destroyAllWindows()
    #text to image
    # summarize what we found
    list=[]
    text=''
    for i in range(len(classIDs)):
       if(labels[classIDs[i] ] not in list):
           list.append(labels[classIDs[i]] )
           text+=labels[classIDs[i] ]
           text+=' '

    
    if(len(list)>0):
        play(text)
if('mp4' in file.lower() or 'avi' in file.lower() ):
    
    cap = cv2.VideoCapture(file)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    name = file.split("/")[-1] if file else 'camera.avi'
    out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('Video file finished.')
            break
        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence,threshold)
        image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors,labels)
        
        cv2.imshow('YOLO Object Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        out.write(image)

    cap.release()
    
    out.release()
    cv2.destroyAllWindows()


