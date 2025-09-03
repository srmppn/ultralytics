import torch
import torch.nn.functional as F

def scale_with_padding(img, scale):
    b, c, h, w = img.shape

    y_coords = torch.linspace(-1, 1, h, device=img.device)
    x_coords = torch.linspace(-1, 1, w, device=img.device)

    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # grid_y = grid_y / scale
    # grid_x = grid_x / scale

    # print('check grid', grid_y)
    # print('check grid', grid_x)
    # Stack and reshape for grid_sample
    # grid_sample expects (B, H, W, 2) where last dim is [x, y]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    scale = scale.view(b, 1, 1, 1)  # [B,1,1,1]
    grid = grid / scale  # shrinks grid coordinates

    # Sample using grid_sample
    scaled_image = F.grid_sample(img, grid, mode='bilinear')
    return scaled_image
    return scaled_image
    # resize_h, resize_w = int(round(h * scale)), int(round(w * scale))
    #
    # if resize_h == 0 or resize_w == 0:
    #     return torch.zeros((b, c, h, w), dtype=img.dtype, device=img.device)
    #
    # # Compute padding
    # pad_h = h - resize_h
    # pad_w = w - resize_w
    #
    # # Divide padding into 2 sides
    # pad_top = pad_h // 2
    # pad_bottom = pad_h - pad_top
    # pad_left = pad_w // 2
    # pad_right = pad_w - pad_left
    #
    # img_resized = F.interpolate(img, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    # return F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=114/255.0)

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