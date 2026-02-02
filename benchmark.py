from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
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
        default="uavdt_complete_datasets/data.yaml",
        help="Path to the dataset .yaml file or directory"
    )
    parser.add_argument(
        '--weight',
        type=str,
        default="uavdt_complete_datasets/data.yaml",
        help="Path to the dataset .yaml file or directory"
    )

    args = parser.parse_args()

    benchmark(
        model=args.weight,   # path to your model file
        data=args.data,     # path to your dataset config
        imgsz=640,             # image size
        half=True,            # use FP16 (True for faster GPU inference)
        int8=False,            # use INT8 (True for edge device optimization)
        device=0               # 0 for GPU, 'cpu' for CPU
    )

