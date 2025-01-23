import os
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import gc

from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.restormer_arch import Restormer
from basicsr.archs.promptir_arch import PromptIR
from basicsr.data.transforms import center_crop

lr_features_1 = []
lr_features_2 = []
lr_features_3 = []
lr_features_4 = []
lr_features_5 = []
lr_features_6 = []
lr_labels = []


def pre_test(img, window_size=8):
    h, w = img.shape[2:]
    mod_pad_h, mod_pad_w = 0, 0
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    img = torch.nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    return img


def generate_features(model, degrad, label_id, window_size=8):
    global lr_features_1
    global lr_features_2
    global lr_features_3
    global lr_features_4
    global lr_features_5
    global lr_features_6
    global lr_labels
    # dehaze: 1
    print(f"begin {degrad}")
    lr_paths = data_paths[f"{degrad}_lr"]
    lr_paths_ = os.listdir(lr_paths)
    for i in trange(len(lr_paths_)):
        if i == 100:
            break
        lr_path = lr_paths_[i]
        lr_img_path = os.path.join(lr_paths, lr_path)
        lr_img = cv2.imread(lr_img_path)
        lr_img = center_crop(lr_img, 128)
        lr_img = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
        with torch.no_grad():
            lr_img = pre_test(lr_img, window_size=window_size)
            outputs = model(lr_img)
            f1, f2, f3, f4, f5, f6 = outputs
            assert torch.sum(torch.isnan(f1)) == 0
            assert torch.sum(torch.isnan(f2)) == 0
            assert torch.sum(torch.isnan(f3)) == 0
            assert torch.sum(torch.isnan(f4)) == 0
            assert torch.sum(torch.isnan(f5)) == 0
            assert torch.sum(torch.isnan(f6)) == 0
        lr_features_1.append(f1.reshape(1, -1).detach().cpu().numpy())
        lr_features_2.append(f2.reshape(1, -1).detach().cpu().numpy())
        lr_features_3.append(f3.reshape(1, -1).detach().cpu().numpy())
        lr_features_4.append(f4.reshape(1, -1).detach().cpu().numpy())
        lr_features_5.append(f5.reshape(1, -1).detach().cpu().numpy())
        lr_features_6.append(f6.reshape(1, -1).detach().cpu().numpy())
        lr_labels.append(label_id)


data_paths = {
    "dehaze_lr": "/public/liguoqi/jkhu/59_jkhu/jkhu/t-SNE/dehaze/hazy", # 1
    "deblur_lr": "/public/liguoqi/jkhu/59_jkhu/jkhu/t-SNE/deblur/HIDE/test/input", # 2
    "denoise_lr": "/public/liguoqi/jkhu/59_jkhu/jkhu/t-SNE/denoise/val/SIDD/input_crops", # 3
    "derain_lr": "/public/liguoqi/jkhu/59_jkhu/jkhu/t-SNE/derain/test/Rain100L/input", # 4
    "low_light_lr": "/public/liguoqi/jkhu/59_jkhu/jkhu/all_in_one_dataset/LowLightDataset/LOL_syn/low", # 5
}

### hook outputs of NAFNet
# model = NAFNetBaseline(3, 32, 12, [2, 4, 8], [2, 2, 2]).cuda()
# model = Restormer().cuda()
# model = SwinIR().cuda()
model = PromptIR()
def init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=2)
        if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=2)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
model.apply(init)
# state_dict = torch.load("pretrain_models/promptir.pth", map_location="cpu")
# model.load_state_dict(state_dict, strict=True)
model.cuda()

model.eval()

generate_features(model, "dehaze", 1)
generate_features(model, "deblur", 2)
generate_features(model, "denoise", 3)
generate_features(model, "derain", 4)
generate_features(model, "low_light", 5)

np_lr_features_1 = np.concatenate(lr_features_1, axis=0)
np_lr_features_2 = np.concatenate(lr_features_2, axis=0)
np_lr_features_3 = np.concatenate(lr_features_3, axis=0)
np_lr_features_4 = np.concatenate(lr_features_4, axis=0)
np_lr_features_5 = np.concatenate(lr_features_5, axis=0)
np_lr_features_6 = np.concatenate(lr_features_6, axis=0)
np_lr_labels = np.array(lr_labels)

print(np_lr_features_1.shape)
print(np_lr_features_2.shape)
print(np_lr_features_3.shape)
print(np_lr_features_4.shape)
print(np_lr_features_5.shape)
print(np_lr_features_6.shape)
print(np_lr_labels.shape)

os.makedirs(f"knns/promptir2", exist_ok=True)

np.save(f"knns/promptir2/lr_features_1.npy", np_lr_features_1)
np.save(f"knns/promptir2/lr_features_2.npy", np_lr_features_2)
np.save(f"knns/promptir2/lr_features_3.npy", np_lr_features_3)
np.save(f"knns/promptir2/lr_features_4.npy", np_lr_features_4)
np.save(f"knns/promptir2/lr_features_5.npy", np_lr_features_5)
np.save(f"knns/promptir2/lr_features_6.npy", np_lr_features_6)
np.save(f"knns/promptir2/lr_labels.npy", np_lr_labels)

del lr_features_1
del lr_features_2
del lr_features_3
del lr_features_4
del lr_features_5
del lr_features_6
del lr_labels
gc.collect()
