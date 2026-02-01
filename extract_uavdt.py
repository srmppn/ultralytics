import os
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt

def uavdt_to_yolo(
    parts, img_w, img_h,
    keep_out_view=(1, 2, 3),
    keep_occlusion=(1, 3, 4)):

    (
        frame_idx,
        target_id,
        left,
        top,
        width,
        height,
        out_of_view,
        occlusion,
        category
    ) = parts

    # Recommended filters (common in UAVDT papers)
    if out_of_view not in keep_out_view:
        return None

    if occlusion not in keep_occlusion:
        return None

    CATEGORY_MAP = {1: 0, 2: 1, 3: 2}

    class_id = CATEGORY_MAP.get(category)
    if class_id is None:
        return None

    x_center = (left + width / 2) / img_w
    y_center = (top + height / 2) / img_h
    w_norm = width / img_w
    h_norm = height / img_h

    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def visualize_frame(img_path, bboxes, title=None):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for f, cls, x, y, w, h, *rest in bboxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(cls),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    cv2.imshow('Video Playback', img)


if __name__ == '__main__':

    img_w, img_h = 1024, 540

#     attribute_director_path = 'uavdt_datasets/train/labels'
#     image_dir = 'uavdt_datasets/train/images'
#     all_entries = os.listdir(attribute_director_path)
#
#     file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(attribute_director_path, entry))]
#     for f_name in file_names:
#         with open(os.path.join(attribute_director_path, f_name), "r") as f:
#             lines = f.readlines()
#
#         print('test', f_name)
#         img_path = os.path.join(image_dir, f_name.replace('.txt', '.jpg'))
#         if os.path.exists(img_path):
#             visualize_frame(
#                 img_path,
#                 lines,
#                 title=f"Frame {f_name} (live)"
#             )

    # found M0210_attr.txt
    # found M0603_attr.txt
    # found M0702_attr.txt
    # found M0703_attr.txt
    # found M0901_attr.txt
    # found M1008_attr.txt
    # found M1102_attr.txt

    # Get a list of all files and directories within the specified path
    attribute_director_path = '/Users/somrak.mon/Downloads/M_attr/train'
    attribute_director_path = '/Users/somrak.mon/Downloads/UAV-benchmark-MOTD_v1.0/GT'
    all_entries = os.listdir(attribute_director_path)
    label_dir = '/Users/somrak.mon/Downloads/fellow3'
    image_dir = '/Users/somrak.mon/Downloads/UAV-benchmark-M/M1102'

    # Filter the list to include only files
    file_names = [entry for entry in all_entries if os.path.isfile(os.path.join(attribute_director_path, entry))]

    sorted_file_names = sorted(file_names)
    frame_to_labels = defaultdict(list)

    for attr_file_name in sorted_file_names:
        if attr_file_name != 'M1102_gt_whole.txt':
            continue

        # Read all lines
        with open(os.path.join(attribute_director_path, attr_file_name), "r") as f:
            lines = f.readlines()

        for line in lines:
            ev = list(map(int, line.strip().split(',')))
#             result = uavdt_to_yolo(ev, img_w, img_h)
#             if not result:
#                 continue

            frame_id, *rest = ev
            frame_to_labels[frame_id].append(ev)

#     for frame_id, labels in frame_to_labels.items():
#         label_name = f"img{frame_id:06d}.txt"
#         label_path = os.path.join(label_dir, label_name)
#         with open(label_path, "w") as f:
#             f.write("\n".join(labels))

    for frame_id, labels in frame_to_labels.items():
        image_name = f"M1102_img{frame_id:06d}.jpg"

        img_path = os.path.join(image_dir, image_name)
        if os.path.exists(img_path):
            visualize_frame(
                img_path,
                labels,
                title=f"Frame {image_name} (live)"
            )

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break