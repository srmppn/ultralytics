import torch
import torch.nn.functional as F

def scale_with_padding(img, scale):
    b, c, h, w = img.shape

    y_coords = torch.linspace(-1, 1, h, device=img.device)
    x_coords = torch.linspace(-1, 1, w, device=img.device)

    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    scale = scale.view(b, 1, 1, 1)  # [B,1,1,1]
    grid = grid / scale  # shrinks grid coordinates

    # Sample using grid_sample
    scaled_image = F.grid_sample(img, grid, mode='bilinear')
    return scaled_image

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