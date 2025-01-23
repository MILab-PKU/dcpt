import math
from collections import OrderedDict
from os import path as osp

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.summary_utils import (
    get_model_activation,
    get_model_complexity_info,
    get_model_flops,
)

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        in_channels = opt["network_g"].get("img_channel", 3)
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        h = opt["network_g"].get("h", 128)
        self.print_network(self.net_g, (1, in_channels, h, h))

        self.grad_clip = opt.get("grad_clip", 0)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
                self.opt.get("remove_norm", False),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("ldl_opt"):
            self.cri_ldl = build_loss(train_opt["ldl_opt"]).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if (
            self.cri_pix is None
            and self.cri_perceptual is None
            and self.cri_ldl is None
        ):
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

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
        if "gt" in data:
            self.gt = data["gt"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()

        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        if self.cri_ldl:
            pixel_weight = get_refined_artifact_map(
                self.gt,
                self.output,
                std=self.opt["train"]["ldl_std"],
            )
            l_ldl = pixel_weight * self.cri_ldl(self.output, self.gt)
            l_total += l_ldl.mean()
            loss_dict["l_ldl"] = l_ldl.mean()
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == "v":
                tfnp = v2np[..., ::-1].copy()
            elif op == "h":
                tfnp = v2np[..., ::-1, :].copy()
            elif op == "t":
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in "v", "h", "t":
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], "t")
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], "h")
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], "v")
            out_list[i] = out_list[i].unsqueeze(0)
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0)

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

    def post_test(self):
        _, _, h, w = self.output.size()
        if "window_size" not in self.opt["network_g"]:
            return
        self.output = self.output[
            :,
            :,
            0 : h - self.mod_pad_h * self.scale,
            0 : w - self.mod_pad_w * self.scale,
        ]

    def test_tile(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.lq.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.lq.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt["tile"]["infer_size"])
        tiles_y = math.ceil(height / self.opt["tile"]["infer_size"])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt["tile"]["infer_size"]
                ofs_y = y * self.opt["tile"]["infer_size"]
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt["tile"]["infer_size"], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt["tile"]["infer_size"], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt["tile"]["tile_pad"], 0)
                input_end_x_pad = min(input_end_x + self.opt["tile"]["tile_pad"], width)
                input_start_y_pad = max(input_start_y - self.opt["tile"]["tile_pad"], 0)
                input_end_y_pad = min(
                    input_end_y + self.opt["tile"]["tile_pad"], height
                )

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = self.lq[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                output_tile = None
                try:
                    if hasattr(self, "net_g_ema"):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                        self.net_g.train()
                except RuntimeError as error:
                    raise error

                # output tile area on total image
                output_start_x = input_start_x * self.opt["scale"]
                output_end_x = input_end_x * self.opt["scale"]
                output_start_y = input_start_y * self.opt["scale"]
                output_end_y = input_end_y * self.opt["scale"]

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt[
                    "scale"
                ]
                output_end_x_tile = (
                    output_start_x_tile + input_tile_width * self.opt["scale"]
                )
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt[
                    "scale"
                ]
                output_end_y_tile = (
                    output_start_y_tile + input_tile_height * self.opt["scale"]
                )

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
    ):
        if self.opt["rank"] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp
            )

    def dist_profile(self, dataloader):
        if self.opt["rank"] == 0:
            self.nondist_profile(dataloader)

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, clamp=True
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

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            self.pre_test()
            if "tile" in self.opt:
                self.test_tile()
            elif "ensemble" in self.opt and self.opt["ensemble"]:
                self.test_selfensemble()
            else:
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
            del self.output
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(
                        {
                            "img": visuals["result"],
                            "img2": visuals["gt"],
                        },
                        opt_,
                    )
                if not clamp:
                    try:
                        assert np.isnan(visuals["result"]).sum() == 0
                    except:
                        visuals["result"][np.isnan(visuals["result"])] = 0

                    self.metric_results["mae"] = 255 * np.abs(
                        np.mean(visuals["result"].clip(0, 1) - visuals["gt"])
                    )

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
        if use_pbar:
            pbar.close()

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

    def nondist_profile(self, dataloader, flops=False):
        logger = get_root_logger()

        if flops:
            H, W = 1280 // self.scale, 720 // self.scale
            # warmup
            try:
                logger.info(
                    get_model_complexity_info(
                        self.net_g, (3, H, W), print_per_layer_stat=False
                    )
                )
                logger.info(get_model_activation(self.net_g, (3, H, W)))
                logger.info(
                    get_model_flops(self.net_g, (3, H, W), print_per_layer_stat=False)
                )
            except:
                logger.warning("OOM when testing on (1280, 720).")

        # test time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timings = np.zeros((len(dataloader), 1))
        memorys = np.zeros((len(dataloader), 1))
        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            self.pre_test()
            torch.cuda.reset_peak_memory_stats()
            starter.record()
            if "tile" in self.opt:
                self.test_tile()
            elif "ensemble" in self.opt and self.opt["ensemble"]:
                self.test_selfensemble()
            else:
                self.test()
            ender.record()
            torch.cuda.synchronize()
            memorys[idx] = (
                torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2
            )
            self.post_test()

            curr_time = starter.elapsed_time(ender)
            timings[idx] = curr_time

        logger.info(f"The average test time is {timings.mean()} ms.")
        logger.info(f"The max allocated mem is {memorys.mean()} M.")

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.float().detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

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
