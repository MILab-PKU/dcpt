import os
import random

import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
    paths_from_folder,
    paths_from_lmdb,
)
from basicsr.data.transforms import augment, center_crop, paired_random_crop
from basicsr.utils import DiffJPEG, FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.mosaic_util import mosaic_CFA_Bayer

from .data_util import prctile_norm


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None
        self.gt_size = self.opt.get("gt_size", None)

        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl
            )
        elif "meta_info_file" in self.opt and self.opt["meta_info_file"] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ["lq", "gt"],
                self.opt["meta_info_file"],
                self.filename_tmpl,
            )
        else:
            multi = opt["multi"] if "multi" in opt else False
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ["lq", "gt"],
                self.filename_tmpl,
                multi=multi,
            )

        self.scale = self.opt["scale"]
        self.depth = self.opt.get("depth", 8)
        img_dtype = self.opt.get("dtype", "uint")
        if img_dtype == "uint":
            self.img_dtype = np.uint16
        elif img_dtype == "float":
            self.img_dtype = np.float32

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]["gt_path"]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=self.img_dtype)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(int(h), int(w), int(c))
            if self.float32 and (self.img_dtype != np.float32):
                img_gt = img_gt.astype(np.float32) / 255.0

        lq_path = self.paths[index]["lq_path"]
        if self.decode:
            img_bytes = self.file_client.get(lq_path, "lq")
            img_lq = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(lq_path, "lq")
            img = np.frombuffer(img_bytes, dtype=self.img_dtype)
            h, w, c = img[0:3]
            img_lq = img[3:].reshape(int(h), int(w), int(c))
            if self.float32 and (self.img_dtype != np.float32):
                img_lq = img_lq.astype(np.float32) / 255.0

        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, self.scale, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            img_lq = prctile_norm(img_lq)

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt["phase"] != "train":
            img_gt = img_gt[
                0 : img_lq.shape[0] * self.scale, 0 : img_lq.shape[1] * self.scale, :
            ]

        # BGR to RGB
        if img_gt.shape[-1] == img_lq.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        # HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class MultiPairedImageDataset(PairedImageDataset):
    def __init__(self, opt):
        super().__init__(opt)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            flag = "grayscale"
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        depth = self.opt.get("depth", 8)
        img_gt = imfrombytes(
            img_bytes,
            flag=flag,
            depth=depth,
            float32=not self.opt.get("prctile_norm", False),
        )

        imgs_lq = []
        for lq_name in os.listdir(self.paths[index]["lq_path"]):
            lq_path = os.path.join(self.paths[index]["lq_path"], lq_name)
            img_bytes = self.file_client.get(lq_path, "lq")
            img_lq = imfrombytes(
                img_bytes,
                flag=flag,
                depth=depth,
                float32=not self.opt.get("prctile_norm", False),
            )
            imgs_lq.append(img_lq)

        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, imgs_lq = paired_random_crop(
                img_gt, imgs_lq, self.gt_size, scale, gt_path
            )
            # flip, rotation
            imgs_lq.append(img_gt)
            imgs_lq = augment(imgs_lq, self.opt["use_hflip"], self.opt["use_rot"])
            img_gt = imgs_lq[-1]
            imgs_lq.pop()

        imgs_lq = np.concatenate(imgs_lq, axis=-1)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            imgs_lq = prctile_norm(imgs_lq)

        # color space transform
        # if "color" in self.opt and self.opt["color"] == "y":
        #     img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt["phase"] != "train":
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float()
        imgs_lq = torch.from_numpy(imgs_lq.transpose(2, 0, 1)).float()
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(imgs_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": imgs_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}


@DATASET_REGISTRY.register()
class PairedImageDenoiseDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDenoiseDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.sigma_type = opt["sigma_type"]
        self.sigma_range = opt["sigma_range"]
        assert self.sigma_type in ["constant", "random", "choice"]
        self.gt_size = opt.get("gt_size", 128)
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
            if self.float32:
                img_gt = img_gt.astype(np.float32) / 255.0

        img_lq = img_gt.copy()
        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            img_lq = prctile_norm(img_lq)

        # BGR to RGB
        if img_gt.shape[-1] == img_lq.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        if self.sigma_type == "constant":
            sigma_value = self.sigma_range
        elif self.sigma_type == "random":
            sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
        elif self.sigma_type == "choice":
            sigma_value = random.choice(self.sigma_range)

        if self.opt["phase"] == "train":
            np.random.seed(seed=index)
        else:
            np.random.seed(seed=0)
        # np.random.seed(seed=0)
        img_lq += np.random.normal(0, sigma_value / 255.0, img_lq.shape)

        # HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageJPEGCARDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageJPEGCARDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.q_type = opt["q_type"]
        self.q_range = opt["q_range"]
        assert self.q_type in ["constant", "random", "choice"]
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None
        self.gt_size = opt.get("gt_size", 128)

        # self.jpeger = DiffJPEG(
        #     differentiable=False
        # )  # simulate JPEG compression artifacts

        self.gt_folder = opt["dataroot_gt"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
            if self.float32:
                img_gt = img_gt.astype(np.float32) / 255.0

        img_lq = img_gt.copy()
        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)

        if self.q_type == "constant":
            q_value = self.q_range
        elif self.q_type == "random":
            q_value = random.uniform(self.q_range[0], self.q_range[1])
        elif self.q_type == "choice":
            q_value = random.choice(self.q_range)

        img_lq = (img_lq * 255).round().astype(np.uint8)
        if img_lq.shape[-1] == 1:
            img_lq = img_lq[..., 0]
        params = [cv2.IMWRITE_JPEG_QUALITY, q_value]
        msg = cv2.imencode(".jpg", img_lq, params)[1]
        img_lq = cv2.imdecode(msg, cv2.IMREAD_UNCHANGED)
        if self.float32:
            img_lq = img_lq.astype(np.float32) / 255.0

        # BGR to RGB
        if img_gt.shape[-1] == img_lq.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        # HWC to CHW, numpy to tensor
        if len(img_gt.shape) == 3:
            img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        elif len(img_gt.shape) == 2:
            img_gt = torch.from_numpy(img_gt).float().contiguous().unsqueeze(0)

        if len(img_lq.shape) == 3:
            img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()
        elif len(img_lq.shape) == 2:
            img_lq = torch.from_numpy(img_lq).float().contiguous().unsqueeze(0)

        # with torch.no_grad():
        #     img_lq = self.jpeger(img_lq.unsqueeze(0), quality=q_value).squeeze(0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDehazeDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDehazeDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.gt_size = opt.get("gt_size", 128)
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            raise NotImplementedError
        else:
            self.paths = paths_from_folder(self.lq_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

        self.suffix = self.opt.get("suffix", ".jpg")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        lq_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(lq_path, "lq")
            img_lq = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(lq_path, "lq")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_lq = img[3:].reshape(h, w, c)
            if self.float32:
                img_lq = img_lq.astype(np.float32) / 255.0

        gt_name = lq_path.split("/")[-1].split("_")[0] + self.suffix
        gt_path = os.path.join(self.gt_folder, gt_name)
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
            if self.float32:
                img_gt = img_gt.astype(np.float32) / 255.0

        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            img_lq = prctile_norm(img_lq)

        # BGR to RGB
        if img_gt.shape[-1] == img_lq.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        # HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageMosaicDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageMosaicDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.gt_size = opt.get("gt_size", 128)
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=False,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
        
        # BGR to RGB first
        if img_gt.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)

        CFA = mosaic_CFA_Bayer(img_lq)[0]
        # NOTE(hujiakui): not matlab here
        img_lq = cv2.cvtColor(CFA, cv2.COLOR_BAYER_BG2BGR_EA)

        # NOTE(hujiakui):  rhos are prior, no need here.
        # rhos, _ = get_rho_sigma(0.255 / 255., iter_num=iter_num)
        # img_lq = (mosaic + rhos.float() * img_lq).div(mask + rhos)

        if self.float32:
            img_gt = img_gt.astype(np.float32) / 255.0
            img_lq = img_lq.astype(np.float32) / 255.0

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            img_lq = prctile_norm(img_lq)

        # HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageInpaintingDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageInpaintingDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.gt_size = opt.get("gt_size", 128)
        self.center_crop = opt["center_crop"] if "center_crop" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=False,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
        
        # BGR to RGB first
        if img_gt.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt["phase"] == "train":
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                img_lq = center_crop(img_lq, self.center_crop)
        
        if self.float32:
            img_gt = img_gt.astype(np.float32) / 255.0
            img_lq = img_lq.astype(np.float32) / 255.0

        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_lq = self.inpainting(img_lq, l_num, l_thick)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            img_lq = prctile_norm(img_lq)

        # HWC to CHW, numpy to tensor
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)
    
    def inpainting(self, img, l_num, l_thick):
        # inpainting
        ori_h, ori_w = img.shape[0], img.shape[1]
        mask = np.zeros((ori_h, ori_w, 3), np.uint8)
        # l_num = random.randint(5, 10)
        # l_thick = random.randint(5, 10)
        col = random.choice(['white', 'black'])
        while (l_num):
            x1, y1 = random.randint(0, ori_w), random.randint(0, ori_h)
            x2, y2 = random.randint(0, ori_w), random.randint(0, ori_h)
            pts = np.array([[x1, y1], [x2, y2]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            mask = cv2.polylines(mask, [pts], 0, (1, 1, 1), l_thick)
            l_num -= 1

        if col == 'white':
            img = np.clip(img + mask, 0, 1)
        else:
            img = np.clip(img - mask, 0, 1)

        return img
