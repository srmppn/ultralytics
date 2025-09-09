import torch
import torch.nn.functional as F
from .ops import clip_boxes

def scale_with_padding(img, scale):
    b, c, h, w = img.shape

    y_coords = torch.linspace(-1, 1, h, device=img.device)
    x_coords = torch.linspace(-1, 1, w, device=img.device)

    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    grid = grid / scale.view(b, 1, 1, 1)

    scaled_image = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    return scaled_image

def scale_boxes_with_padding(boxes, scales, h, w):
    """
    boxes: Tensor (N, 4) with (x_min, y_min, x_max, y_max)
    scale: float
    h, w: original image height, width
    """
    B, M, _ = boxes.shape

    # reshape scales for broadcasting
    scales = scales.view(B, 1, 1)

    # scaled dimensions
    resize_h = (h * scales).round()
    resize_w = (w * scales).round()

    # compute padding
    pad_h = h - resize_h
    pad_w = w - resize_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    # broadcast padding for each box
    pad_top = pad_top.view(B, 1, 1)
    pad_left = pad_left.view(B, 1, 1)

    # clone boxes to avoid modifying in-place
    boxes = boxes.clone()

    # scale and add padding
    boxes[..., [0, 2]] = boxes[..., [0, 2]] * scales + pad_left
    boxes[..., [1, 3]] = boxes[..., [1, 3]] * scales + pad_top
    return clip_boxes(boxes, (h, w))

def scale_boxes_with_padding_inverse(b_size, boxes, scales, h, w):
    """
    boxes: Tensor (N, 4) with (x_min, y_min, x_max, y_max)
    scale: float
    h, w: original image height, width
    """
    # scaled dimensions
    resize_h = (h / scales).round()
    resize_w = (w / scales).round()

    # compute padding
    pad_h = h - resize_h
    pad_w = w - resize_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    # broadcast padding for each box
    pad_top = pad_top
    pad_left = pad_left

    # clone boxes to avoid modifying in-place
    boxes = boxes.clone()

    # scale and add padding
    boxes[..., [0, 2]] -= pad_left
    boxes[..., [1, 3]] -= pad_top

    boxes[..., [0, 2]] = boxes[..., [0, 2]] / scales
    boxes[..., [1, 3]] = boxes[..., [1, 3]] / scales

    return clip_boxes(boxes, (h, w))