from collections import OrderedDict

import torch
from timm.utils.metrics import accuracy
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class DCPTModel(BaseModel):
    """Base degradation classification model for image restoration."""

    def __init__(self, opt):
        super(DCPTModel, self).__init__(opt)

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
        self.net_g.train()
        self.net_dc.train()
        train_opt = self.opt["train"]

        # hook net_g here
        self.hook_outputs = list()

        self.hooks = list()
        hook_names = self.opt.get("hook_names", None)
        for name, module in self.net_g.named_modules():
            if hook_names in name and name.count(".") == 1:
                hook = module.register_forward_hook(self.hook_forward_fn)
                self.hooks.append(hook)

        # define losses
        if train_opt.get("classify_opt"):
            self.cri_classify = build_loss(train_opt["classify_opt"]).to(self.device)
        else:
            self.cri_classify = None

        if train_opt.get("pixel_opt"):
            self.cri_pixel = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pixel = None

        if self.cri_classify is None:
            raise ValueError("Classify loss is None.")

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

        optim_params = []
        for k, v in self.net_dc.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_dc"].pop("type")
        self.optimizer_dc = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_dc"]
        )

        self.optimizers.append(self.optimizer_dc)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.dataset_idx = data["dataset_idx"].to(self.device, non_blocking=True)
        # self.dataset_idx = F.one_hot(self.dataset_idx)
        if "gt" in data:
            self.gt = data["gt"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        ### train to generate the clean image
        self.net_g.train()
        self.net_dc.eval()
        self.optimizer_g.zero_grad()
        pix_output = self.net_g(self.gt, hook=False)
        self.hook_outputs = list()

        l_total = 0
        # pixel loss
        if self.cri_pixel:
            l_pix = self.cri_pixel(pix_output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        ### train to classify the degradation
        self.net_dc.train()
        self.optimizer_dc.zero_grad()

        self.net_g(self.lq, hook=True)
        cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])

        # classify loss
        if self.cri_classify:
            l_classify = self.cri_classify(cls_output, self.dataset_idx)
            l_total += l_classify
            loss_dict["l_classify"] = l_classify

        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_dc.step()

        self.hook_outputs = list()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_network(self.net_dc, "net_dc", current_iter)
        self.save_training_state(epoch, current_iter)

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
        self.net_g(self.lq, hook=True)

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img=False, clamp=True
    ):
        if self.opt["rank"] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, clamp
            )

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img=False, clamp=True
    ):
        with_metrics = True
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {"top-1": 0.0}
            self._initialize_best_metric_results()
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            self.pre_test()
            self.test()

            output = self.net_dc(self.lq, self.hook_outputs[::-1])

            if with_metrics:
                # calculate metrics
                self.metric_results["top-1"] += accuracy(output, self.dataset_idx)[0]

            self.hook_outputs = []
            del output
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {idx}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    metric, self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, tb_logger)

    def _initialize_best_metric_results(self):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        # add a dataset record
        record = dict()
        record["top-1"] = dict(val=0.0, iter=-1)
        self.best_metric_results = record

    def _update_best_metric_result(self, metric, val, current_iter):
        if val >= self.best_metric_results[metric]["val"]:
            self.best_metric_results[metric]["val"] = val
            self.best_metric_results[metric]["iter"] = current_iter

    def _log_validation_metric_values(self, current_iter, tb_logger):
        log_str = f"Validation Degradation Classifier.\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)
