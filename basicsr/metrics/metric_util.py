import numpy as np

from basicsr.utils import bgr2ycbcr


def reorder_image(img, input_order="BHWC"):
    """Reorder images to 'BHWC' order.

    If the input_order is (h, w), return (1, h, w, 1);
    If the input_order is (c, h, w), return (1, h, w, c);
    If the input_order is (h, w, c), return (1, h, w, c);
    If the input_order is (b, h, w, c), return (b, h, w, c);
    If the input_order is (b, c, h, w), return (b, h, w, c);

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'BHWC' or 'BCHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'BHWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
        img = img[None, ...]
    if input_order == "BCHW":
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
            img = img[None, ...]
        elif len(img.shape) == 4:
            img = img.transpose(0, 2, 3, 1)
    return img


def to_y_channel(img, image_range=255.0):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / image_range
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * image_range
