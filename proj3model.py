import torch
from ultralytics import YOLO
from PIL import Image


# Step 2: YOLOv8 Training
# ----------------------------------------------
from ultralytics import YOLO

# Load the YOLOv8 model

model = YOLO("yolov8n.pt")


# Use the model

model.train(data="/home/haque/Project 3/data/data.yaml"
            , epochs=500, imgsz = 1200, batch = 2, name = 'final_model')  # train the model

metrics = model.val()  # evaluate model performance on the validation set

# Step 2: Evaluation
# ----------------------------------------------
# Evaluate the model
for image_path in ['/home/haque/Project 3/data/evaluation/ardmega.jpg',
                    '/home/haque/Project 3/data/evaluation/arduno.jpg',
                    '/home/haque/Project 3/data/evaluation/rasppi.jpg']:
    img = Image.open(image_path)
    results = model.predict(img, save=True)
    print(results)  # print results to stdout