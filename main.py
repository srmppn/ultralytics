from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    yolo = YOLO("runs/detect/train56/weights/best.pt")
    results = yolo.predict("sandbox/test_3.png")
    annotated_frame = results[0].plot()
    cv2.imshow("Detections", annotated_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()