from ultralytics import YOLO
import torch.nn as nn
import argparse

import cv2
import torch

if __name__ == '__main__':
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="YOLO Adaptive Resize Training")
    parser.add_argument(
        '--data',
        type=str,
        default="/content/drive/MyDrive/uavdt_complete_datasets/data.yaml",
        help="Path to the dataset .yaml file or directory"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--weight',
        type=float,
        default=1.2,
        help="Initial weight"
    )

    args = parser.parse_args()

    yolo = YOLO("ultralytics/cfg/models/11/yolo11-adaptive-resize.yaml").load("yolo11n.pt")

    with torch.no_grad():
        weight = args.weight
        # You may need to verify the exact path to adaptive_resize
        # Use print(model.model) to see the structure if this path fails
        target_layer = yolo.model.model[-2].adaptive_resize[0]
        nn.init.constant_(target_layer.weight, weight)
        print(f"Successfully reset adaptive_resize weights to {weight}")

    yolo.train(data=args.data, epochs=args.epochs, imgsz=640, mosaic=0, scale=0)

    with torch.no_grad():
        # You may need to verify the exact path to adaptive_resize
        # Use print(model.model) to see the structure if this path fails
        target_layer = yolo.model.model[-2].adaptive_resize[0]
        print(f"Check result {target_layer.weight}")