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

    yolo = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("yolo11n.pt")

    yolo.train(data=args.data, epochs=args.epochs, imgsz=640, mosaic=0, scale=0)
