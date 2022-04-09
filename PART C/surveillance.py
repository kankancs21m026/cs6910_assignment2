from ast import arg
import numpy as np
import argparse
import cv2
import os
import time
from textToSpeech import play
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from  videoUtility import extract_boxes_confidences_classids,draw_bounding_boxes,make_prediction
from alertSMS import SendMessage
import sys
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

if('mp4' in file.lower() or 'avi' in file.lower() ):
    
    cap = cv2.VideoCapture(file)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    name = file.split("/")[-1] if file else 'camera.avi'
    out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    unauthorizedPersonCount=0
    messagesent=0
    while cap.isOpened():
        ret, image = cap.read()
        imgwithoutbox=image
        if not ret:
            print('Video file finished.')
            break
        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence,threshold)
        markLabels=['bus','car','bicycle','person']
       
        image,unauthorizedPersonCount = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors,labels,markLabels,unauthorizedPersonCount,type='monitor')
        
        #send message if unauthorized person roming arround the restricted area
        if(unauthorizedPersonCount>50 and messagesent!=1):
            SendMessage()
           
            filename = 'output.jpg'
            # Saving the image
            cv2.imwrite(filename, imgwithoutbox)
            messagesent=1
        
        cv2.imshow('YOLO Object Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        out.write(image)

    cap.release()
    
    out.release()
    cv2.destroyAllWindows()


