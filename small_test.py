import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from ultralytics import YOLO
import torch
device = torch.device("cuda:3")

model = YOLO('./yolov8n.pt')

def train_driver_hand():

    data_path = r"/data/clearingvehicle/eating/hand_traindata_v1/data.yaml"

    model.train(
        data=data_path,
        epochs=150,
        imgsz=1080,
        batch=16,
        device=0,

        project="model",
        name="run1",
        exist_ok=True,
        verbose=True
    )


if __name__ == "__main__":
    train_driver_hand()

