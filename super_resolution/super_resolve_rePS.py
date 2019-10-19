from __future__ import print_function
import os
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn as nn
from time import time
from math import log10
import numpy as np
import torch.onnx

def psnr_evaluate(out_img, target):
    criterion = nn.MSELoss()
    img_to_tensor = ToTensor()
    target = img_to_tensor(target)
    out_img = img_to_tensor(out_img)
    mse = criterion(out_img, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to be inferenced')
parser.add_argument('--model', type=str, required=True, help='.pth model file to be used')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--psnr_target', help='optional, evaluate psnr with target image', type=str, default=None)
parser.add_argument('-c', '--compare', help='optional, show psnr comparision of build-in resize algorithm', default=False, action="store_true")
parser.add_argument('--onnx_export', help='optional, export ONNX model', type=str, default=None)
parser.add_argument('-u', '--upscale_factor', help='set upscale factor', default=2, type=int)
opt = parser.parse_args()
img = Image.open(opt.input_image).convert('YCbCr')
y, cb, cr = img.split()

img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if opt.cuda:
    model = torch.load(opt.model)
    model = model.cuda()
    input = input.cuda()
else:
    model = torch.load(opt.model, map_location=lambda storage, loc: storage)
    model = model.cpu()
    input = input.cpu()

model.eval()
out = model(input)
out = out.cpu()

out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.output_filename)
print('Output image saved to', opt.output_filename)
print(' ')

if opt.psnr_target is not None:
    target = Image.open(opt.psnr_target).convert('RGB')
    print('PSNR evaluation:')

    if opt.compare:
        test = Image.open(opt.input_image).convert('RGB')
        width, height = test.size

        nn_test = test.resize((int(width*opt.upscale_factor), int(height*opt.upscale_factor)) )#, Image.ANTIALIAS)
        print('resize with PIL NEAREST   : {:.2f}'.format(psnr_evaluate(nn_test, target)))

        bili_test = test.resize((int(width*opt.upscale_factor), int(height*opt.upscale_factor)), Image.BILINEAR)
        print('resize with PIL BILINEAR  : {:.2f}'.format(psnr_evaluate(bili_test, target)))

        bicu_test = test.resize((int(width*opt.upscale_factor), int(height*opt.upscale_factor)), Image.BICUBIC)
        print('resize with PIL BICUBIC   : {:.2f}'.format(psnr_evaluate(bicu_test, target)))

        anti_test = test.resize((int(width*opt.upscale_factor), int(height*opt.upscale_factor)), Image.ANTIALIAS)
        print('resize with PIL ANTIALIAS : {:.2f}'.format(psnr_evaluate(anti_test, target)))

    print('super resolution PIL      : {:.2f}\n'.format(psnr_evaluate(out_img, target)))

    print('Delta evaluation (4 bits):')
    outnumpy = np.array(out_img).astype('int16')
    tarnumpy = np.array(target).astype('int16')
    delta = tarnumpy-outnumpy
    absdelta = np.abs(delta)
    print('average difference (abs): {:.2f}'.format(np.average(absdelta)))
    print('delta value > 8         : {} pixels'.format((delta > 8).sum()))
    print('delta value < -7        : {} pixels'.format((delta < -7).sum()))
    print('maximum delta value     : {} '.format(np.amax(delta)))
    print('minimum delta value     : {} '.format(np.amin(delta)))
    print('delta shape             : ' + str(delta.shape))

print(input.shape)
if opt.onnx_export is not None: # input images shape should be fixed value
    print(model)
    test_a = list(model.children())
    test_b = [test_a[1], test_a[0], test_a[2], test_a[0], test_a[3], test_a[0], test_a[4]]
    model = torch.nn.Sequential(*test_b)
    print(test_b)
    print(model)
    torch.onnx.export(model, input, opt.onnx_export, input_names = ['input'], output_names = ['output'])
