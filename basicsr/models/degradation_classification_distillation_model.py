from collections import OrderedDict
from os import path as osp

import cv2
import numpy as np
import torch
from timm.utils.metrics import accuracy
from torch.nn import functional as F
from tqdm import tqdm
import torchvision

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.data.transforms import center_crop

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class DCDistModel(BaseModel):
    """Degradation classification based Distillation model for image restoration."""

    def __init__(self, opt):
        super(DCDistModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)

        self.net_dc = build_network(opt["network_dc"])
        self.net_dc = self.model_to_device(self.net_dc)

        # load pretrained models
        load_path_g = self.opt["path"].get("pretrain_network_g", None)
        load_path_dc = self.opt["path"].get("pretrain_network_dc", None)
        if load_path_g is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path_g,
                self.opt["path"].get("strict_load_g", True),
                param_key,
                self.opt.get("remove_norm", False),
            )
        if load_path_dc is not None:
            param_key = self.opt["path"].get("param_key_dc", "params")
            self.load_network(
                self.net_dc,
                load_path_dc,
                self.opt["path"].get("strict_load_dc", True),
                param_key,
                self.opt.get("remove_norm", False),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_dc.eval()
        for param in self.net_dc.parameters():
            param.requires_grad = False

        self.net_g.train()
        train_opt = self.opt["train"]

        self.grad_clip = self.opt.get("grad_clip", 0)

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # hook net_g here
        self.hook_outputs = list()

        self.hooks = list()
        hook_names = self.opt.get("hook_names", None)
        for name, module in self.net_g.named_modules():
            if hook_names in name and name.count(".") == 1:
                # For Restormer_origin
                pre = name.split(".")[0]
                if (int(name.split(".")[-1]) == 5 and int(pre[-1]) in [2, 3]) or (int(name.split(".")[-1]) == 3 and int(pre[-1]) == 1):
                    hook = module.register_forward_hook(self.hook_forward_fn)
                    self.hooks.append(hook)

        if self.ema_decay > 0:
            for name, module in self.net_g_ema.named_modules():
                if hook_names in name:
                    hook = module.register_forward_hook(self.hook_forward_fn)
                    self.hooks.append(hook)

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pixel = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pixel = None

        if train_opt.get("classify_opt"):
            self.cri_classify = build_loss(train_opt["classify_opt"]).to(self.device)
        else:
            self.cri_classify = None

        if self.cri_classify is None and self.cri_pixel is None:
            raise ValueError("Classify loss and Pixel loss are both None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def hook_forward_fn(self, module, input, output):  # noqa
        if isinstance(output, tuple):
            output = output[-1]
        self.hook_outputs.append(output)

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        if "dataset_idx" in data.keys():
            self.dataset_idx = data["dataset_idx"].to(self.device, non_blocking=True)
        if "dataset_idx" in self.opt:
            dataset_idx = self.opt.get("dataset_idx", None)
            batch = self.lq.shape[0]
            self.dataset_idx = torch.ones((batch, )).type_as(self.lq).long() * dataset_idx
        # self.dataset_idx = F.one_hot(self.dataset_idx)
        if "gt" in data:
            self.gt = data["gt"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.net_dc.eval()
        self.net_g.train()
        self.optimizer_g.zero_grad()

        self.pix_output = self.net_g(self.lq)
        self.cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pixel:
            l_pixel = self.cri_pixel(self.pix_output, self.gt)
            l_total += l_pixel
            loss_dict["l_pixel"] = l_pixel
        # cls loss
        if self.cri_classify:
            l_classify = self.cri_classify(self.cls_output, self.dataset_idx)
            l_total += l_classify
            loss_dict["l_classify"] = l_classify

        l_total.backward()

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)

        self.optimizer_g.step()

        self.hook_outputs = list()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def check_window_size(self, window_size_stats):
        window_size, stats = window_size_stats
        if not (
            isinstance(window_size, tuple)
            or isinstance(window_size, list)
            and not stats
        ):
            return [window_size, True]
        return self.check_window_size([max(window_size), False])

    def pre_test(self):
        # pad to multiplication of window_size
        _, _, h, w = self.lq.size()
        if "window_size" not in self.opt["network_g"]:
            return

        # FIXME: this is only supported when the shape of lq's H == W
        window_size, _ = self.check_window_size(
            [self.opt["network_g"].get("window_size", h), False]
        )
        self.scale = self.opt.get("scale", 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.lq = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    @torch.no_grad()
    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.pix_output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.pix_output = self.net_g(self.lq)
            self.net_g.train()

        self.lq = torchvision.transforms.functional.center_crop(self.lq, 128)
        # self.cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])

    def post_test(self):
        _, _, h, w = self.pix_output.size()
        if "window_size" not in self.opt["network_g"]:
            return
        self.pix_output = self.pix_output[
            :,
            :,
            0 : h - self.mod_pad_h * self.scale,
            0 : w - self.mod_pad_w * self.scale,
        ]

    def validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img=False,
        clamp=True,
        dataset_idx=0,
    ):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt["dist"]:
            self.dist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp, dataset_idx
            )
        else:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp, dataset_idx
            )

    def dist_validation(
        self,
        dataloader,
        current_iter,
        tb_logger,
        save_img=False,
        clamp=True,
        dataset_idx=0,
    ):
        if self.opt["rank"] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp, dataset_idx
            )

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True, dataset_idx=0
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        # logger = get_root_logger()

        # cls_acc = 0.0
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            self.pre_test()
            self.test()
            self.post_test()

            visuals = self.get_current_visuals()
            if clamp:
                visuals["result"] = visuals["result"].clamp(0, 1)
                visuals["gt"] = visuals["gt"].clamp(0, 1)
            visuals["result"] = visuals["result"].numpy()
            visuals["gt"] = visuals["gt"].numpy()

            del self.gt
            del self.lq
            del self.pix_output
            self.hook_outputs = list()

            if with_metrics:
                # calculate metrics
                for i, img_path in enumerate(val_data["lq_path"]):
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    # log_str = f"\n\t{img_name}: \n"
                    for name, opt_ in self.opt["val"]["metrics"].items():
                        metric = calculate_metric(
                            {
                                "img": visuals["result"],
                                "img2": visuals["gt"],
                            },
                            opt_,
                        )
                        self.metric_results[name] += metric
                        # log_str += f"\t # {name}: {metric:.4f}"
                        # log_str += "\n"
                    # cls_acc += accuracy(
                    #     self.cls_output,
                    #     torch.tensor(dataset_idx).type_as(self.cls_output).unsqueeze(0),
                    # )[0]
                    # cls_acc = accuracy(self.cls_output, torch.tensor(dataset_idx).type_as(self.cls_output).unsqueeze(0))[0]
                    # log_str += f"\t # degradation classification acc: {cls_acc:.2f}"
                    # log_str += "\n"
                    # logger.info(log_str)

            # del self.cls_output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if save_img:
                for i, img_path in enumerate(val_data["lq_path"]):
                    depth = self.opt["depth"] if "depth" in self.opt else 8
                    if depth == 16:
                        sr_img = (
                            (visuals["result"][i, ...] * 65535.0)
                            .round()
                            .astype(np.uint16)
                        )
                    else:
                        sr_img = (
                            (visuals["result"][i, ...] * 255.0).round().astype(np.uint8)
                        )
                    if sr_img.shape[0] == 3:
                        sr_img = cv2.cvtColor(
                            sr_img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR
                        )
                    if sr_img.shape[-1] == 1:
                        sr_img = sr_img[..., 0]
                    img_name = osp.splitext(osp.basename(img_path))[0]
                    if self.opt["is_train"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )
                    else:
                        if self.opt["val"]["suffix"]:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}.png',
                            )
                        else:
                            save_img_path = osp.join(
                                self.opt["path"]["visualization"],
                                dataset_name,
                                f'{img_name}_{self.opt["name"]}.png',
                            )
                    imwrite(sr_img, save_img_path, depth=depth)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                if clamp:
                    self._update_best_metric_result(
                        dataset_name, metric, self.metric_results[metric], current_iter
                    )
            if clamp:
                self._log_validation_metric_values(
                    current_iter, dataset_name, tb_logger
                )

            # cls_acc /= idx + 1
            # log_str = f"degradation classification acc: {cls_acc:.2f}"
            # log_str += "\n"
            # logger.info(log_str)
            if use_pbar:
                pbar.close()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.pix_output.float().detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict
