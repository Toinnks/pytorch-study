from ultralytics import YOLO
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# Load a model
model = YOLO('./yolov8n.pt') 
# Train the model
results = model.train(data='./data.yaml', epochs=160,  batch=32, workers=0)  

#  degrees=10,scale=0.5,flipud=0.3, mosaic=1.0, mixup=0.5