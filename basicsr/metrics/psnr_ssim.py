import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    psnrs = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
            img = to_y_channel(img, image_range)
            img2 = to_y_channel(img2, image_range)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img - img2) ** 2)
        if mse == 0:
            return float("inf")
        psnrs.append(10.0 * np.log10(image_range * image_range / mse))

    return np.array(psnrs).mean()


@METRIC_REGISTRY.register()
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel and img.shape[1] == img2.shape[1] == 3:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


@METRIC_REGISTRY.register()
def calculate_ssim(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    ssims = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
            img = to_y_channel(img, image_range)
            img2 = to_y_channel(img2, image_range)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)

        for j in range(img.shape[2]):
            ssim, _ = _ssim(img[..., j], img2[..., j], image_range)
            ssims.append(ssim)
    return np.array(ssims).mean()


def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)


def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())


def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)


class SSIM_MATLAB_Pytorch:
    def __init__(self, device='cpu'):
        self.device = device

        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)

    def __call__(self, img1, img2, image_range):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            if len(img1.shape) == 2:
                conv = self.conv2d
            elif len(img1.shape) == 3:
                conv = self.conv3d
            else:
                raise not NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv, image_range)

    def _ssim(self, img1, img2, conv, image_range):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * image_range) ** 2
        C2 = (0.03 * image_range) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()


@METRIC_REGISTRY.register()
def calculate_ssim_matlab(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """
    ssim_calc = SSIM_MATLAB_Pytorch()

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    ssims = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
            img = to_y_channel(img, image_range)
            img2 = to_y_channel(img2, image_range)

        # img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        # img2 = torch.from_numpy(img2).permute(2, 0, 1).contiguous()
        img = torch.from_numpy(img).float()
        img2 = torch.from_numpy(img2).float()

        for j in range(img.shape[2]):
            ssim = ssim_calc(img[..., j], img2[..., j], image_range)
            ssims.append(ssim)
        # ssim = ssim_calc(img, img2, image_range)
        ssims.append(ssim)

    del ssim_calc
    return np.array(ssims).mean()


@METRIC_REGISTRY.register()
def calculate_msssim(
    img,
    img2,
    crop_border,
    weights=None,
    image_range=255,
    input_order="BCHW",
    test_y_channel=False,
    **kwargs,
):
    """Calculate MS-SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    For three-channel images, MS-SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'BHWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: MS-SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    batch = imgs.shape[0]
    results = []
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
            img = to_y_channel(img, image_range)
            img2 = to_y_channel(img2, image_range)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)
        downsample_filter = np.ones((2, 2)) / 4

        ssims = []
        css = []
        level = len(weights)
        for _ in range(level):
            for j in range(img.shape[2]):
                ssim, cs = _ssim(img[..., j], img2[..., j], image_range)
                ssims.append(ssim)
                css.append(cs)
                img = cv2.filter2D(
                    img,
                    -1,
                    downsample_filter,
                    anchor=(0, 0),
                    borderType=cv2.BORDER_REFLECT,
                )
                img2 = cv2.filter2D(
                    img2,
                    -1,
                    downsample_filter,
                    anchor=(0, 0),
                    borderType=cv2.BORDER_REFLECT,
                )
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]
                    img2 = img2[..., np.newaxis]
        result = np.prod(np.power(css[: level - 1], weights[: level - 1])) * (
            ssims[level - 1] ** weights[level - 1]
        )
        results.append(result)

    return np.array(results).mean()


@METRIC_REGISTRY.register()
def calculate_ssim_pt(
    img,
    img2,
    crop_border,
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate SSIM (structural similarity) (PyTorch version).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel and img.shape[1] == img2.shape[1] == 3:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim, _ = _ssim_pth(img, img2, image_range)
    return ssim


def _ssim(img, img2, image_range=255):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * image_range) ** 2
    c2 = (0.03 * image_range) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(), cs_map.mean()


def _ssim_pth(img, img2, image_range=1.0):
    """Calculate SSIM (structural similarity) (PyTorch version).

    It is called by func:`calculate_ssim_pt`.

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).

    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * image_range) ** 2
    c2 = (0.03 * image_range) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = (
        torch.from_numpy(window)
        .view(1, 1, 11, 11)
        .expand(img.size(1), 1, 11, 11)
        .to(img.dtype)
        .to(img.device)
    )

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(
        img2, window, stride=1, padding=0, groups=img2.shape[1]
    )  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2
    )

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3]), cs_map.mean([1, 2, 3])


@METRIC_REGISTRY.register()
def calculate_nrmse(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    nrmses = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
            img = to_y_channel(img, image_range)
            img2 = to_y_channel(img2, image_range)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)

        rmse = np.sqrt(np.mean((img - img2) ** 2))
        if rmse == 0:
            return float("inf")
        nrmses.append(rmse / (img.max() - img.min()))

    return np.array(nrmses).mean()
