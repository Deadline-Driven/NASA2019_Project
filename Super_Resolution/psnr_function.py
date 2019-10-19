from math import log10, sqrt
import numpy as np
def psnr_numpy(target, ref):
    target_data = np.array(target, dtype=np.float32)
    target_data /= 255.0
    ref_data = np.array(ref, dtype=np.float32)
    ref_data /= 255.0
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = sqrt( np.mean(diff ** 2.) )
    return 20*log10(1.0/rmse)


import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
def psnr_torch(out_img, target):
    criterion = nn.MSELoss()
    img_to_tensor = ToTensor()
    target = img_to_tensor(target)
    out_img = img_to_tensor(out_img)
    mse = criterion(out_img, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr

# ===========psnr_numpy sample usage==========
# cv2_target = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB)
# cv2_test = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
# print(psnr_numpy(cv2_target, cv2_test))
#
# ===========psnr_torch sample usage==========
# target = Image.open(target_image_path).convert('RGB')
# test = Image.open(test_image_path).convert('RGB')
# print(psnr_torch(target, test))
