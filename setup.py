from ultralytics import YOLO
import torch.nn as nn

import cv2
import torch

if __name__ == '__main__':
    yolo = YOLO("ultralytics/cfg/models/11/yolo11-adaptive-resize.yaml").load("yolo11n.pt")

    with torch.no_grad():
        weight = 1.2
        # You may need to verify the exact path to adaptive_resize
        # Use print(model.model) to see the structure if this path fails
        target_layer = yolo.model.model[-2].adaptive_resize[0]
        nn.init.constant_(target_layer.weight, weight)
        print(f"Successfully reset adaptive_resize weights to {weight}")

    yolo.train(data="uavdt_complete_datasets/data.yaml", epochs=1, imgsz=640, mosaic=0, scale=0)

    with torch.no_grad():
        # You may need to verify the exact path to adaptive_resize
        # Use print(model.model) to see the structure if this path fails
        target_layer = yolo.model.model[-2].adaptive_resize[0]
        print(f"Check result {target_layer.weight}")