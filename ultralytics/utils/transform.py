import torch
import torch.nn.functional as F

def scale_with_padding(img, scale):
    b, c, h, w = img.shape
    resize_h, resize_w = int(round(h * scale)), int(round(w * scale))

    if resize_h == 0 or resize_w == 0:
        return torch.zeros((b, c, h, w), dtype=img.dtype, device=img.device)

    # Compute padding
    pad_h = h - resize_h
    pad_w = w - resize_w

    # Divide padding into 2 sides
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_resized = F.interpolate(img, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    return F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=114/255.0)

def scale_boxes_with_padding(boxes, scale, h, w):
    """
    boxes: Tensor (N, 4) with (x_min, y_min, x_max, y_max)
    scale: float
    h, w: original image height, width
    """
    # scaled dimensions
    resize_h, resize_w = int(round(h * scale)), int(round(w * scale))

    # compute padding (same as in scale_with_padding)
    pad_h = h - resize_h
    pad_w = w - resize_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    # scale coordinates
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
    return boxes