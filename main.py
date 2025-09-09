from ultralytics import YOLO

import cv2
import torch

from ultralytics.utils.transform import scale_boxes_with_padding, scale_boxes_with_padding_inverse

if __name__ == '__main__':
    yolo = YOLO("research/best_8.pt")#.load("sandbox/yolov8n.pt")
    # train
    # yolo.train(data="sandbox/train_model_3/data.yaml", epochs=100, imgsz=640, mosaic=0, scale=0)

    # predict
    results = yolo.predict("sandbox/train_model_3/valid/images/test_mp4-0033_jpg.rf.d07b3998a1ec68f22c21768c35b5f7a3.jpg", conf=0.5, iou=0.3)
    annotated_frame = results[0].plot(labels=False)

    cv2.imshow("Detections", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # benchmark
    # yolo.benchmark(data="sandbox/train_model_3/data.yaml")

    # result = scale_boxes_with_padding(torch.tensor([[[10, 10, 10, 10]]]), torch.tensor([2]), 1, 4)
    #
    # print(result)
    #
    # result = scale_boxes_with_padding_inverse(1, result, torch.tensor([2]), 4, 1)
    #
    # print(result)
