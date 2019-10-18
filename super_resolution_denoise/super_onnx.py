from __future__ import print_function
import argparse
from time import time
from math import log10, sqrt
import numpy as np
import onnxruntime
import cv2

# Image generated from different library will cause a little difference in pixel value, but it's ok

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float32)
    target_data /= 255.0
    ref_data = np.array(ref, dtype=np.float32)
    ref_data /= 255.0
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = sqrt( np.mean(diff ** 2.) )
    return 20*log10(1.0/rmse)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to be inferenced')
parser.add_argument('--model', type=str, required=True, help='onnx model file to be used')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
parser.add_argument('--psnr_target', help='optional, evaluate psnr with target image', type=str, default=None)
parser.add_argument('-c', '--compare', help='optional, show psnr of build-in resize algorithm', default=False, action="store_true")
parser.add_argument('-u', '--upscale_factor', help='set upscale factor', default=2, type=int)
opt = parser.parse_args()

# ==== Inference and preprocessing start ====

cv2img = cv2.imread(opt.input_image)
cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2YCR_CB)
(cv2_y, cv2_cr, cv2_cb) = cv2.split(cv2img)
cv2_y = cv2_y.astype(np.float32)
cv2_y /= 255.0
cv2_y = np.expand_dims(cv2_y, 0)
cv2_y = np.expand_dims(cv2_y, 0)

ort_session = onnxruntime.InferenceSession(opt.model)
ort_inputs = {ort_session.get_inputs()[0].name: cv2_y}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

cv2_out_img_y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
cv2_out_img_cb = cv2.resize(cv2_cb, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img_cr = cv2.resize(cv2_cr, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img =  cv2.merge([cv2_out_img_y, cv2_out_img_cr, cv2_out_img_cb])
cv2_out_img = cv2.cvtColor(cv2_out_img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(opt.output_filename, cv2_out_img)

# ==== Inference and postprocessing end ====

print('Output image saved to', opt.output_filename)
print(' ')

if opt.psnr_target is not None:
    cv2_target = cv2.cvtColor(cv2.imread(opt.psnr_target), cv2.COLOR_BGR2RGB)
    print('PSNR evaluation:')

    if opt.compare:
        cv2_test_img = cv2.cvtColor(cv2.imread(opt.input_image), cv2.COLOR_BGR2RGB)

        nn_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
        print('resize with OpenCV INTER_NEAREST : {:.2f}'.format(psnr(nn_img, cv2_target)))

        bili_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
        print('resize with OpenCV INTER_LINEAR  : {:.2f}'.format(psnr(bili_img, cv2_target)))

        area_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_AREA)
        print('resize with OpenCV INTER_AREA    : {:.2f}'.format(psnr(area_img, cv2_target)))

        bicu_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        print('resize with OpenCV INTER_CUBIC   : {:.2f}'.format(psnr(bicu_img, cv2_target)))

        lanc_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_LANCZOS4)
        print('resize with OpenCV INTER_LANCZOS4: {:.2f}'.format(psnr(lanc_img, cv2_target)))

    cv2_pyt_out_img = cv2.cvtColor(cv2.imread(opt.output_filename), cv2.COLOR_BGR2RGB)
    print('super resolution OpenCV (python) : {:.2f}'.format(psnr(cv2_pyt_out_img, cv2_target)))

    #cv2_cpp_out_img = cv2.cvtColor(cv2.imread('output_cpp.png'), cv2.COLOR_BGR2RGB)
    #print('super resolution OpenCV (c++)    : {:.2f}'.format(psnr(cv2_cpp_out_img, cv2_target)))

    print('\nDelta evaluation (4 bits):')
    outnumpy = np.array(cv2_pyt_out_img).astype('int16') #image generated from c++ inference program
    tarnumpy = np.array(cv2_target).astype('int16')
    delta = tarnumpy-outnumpy
    absdelta = np.abs(delta)
    print('average difference (abs): {:.2f}'.format(np.average(absdelta)))
    print('delta value > 8         : {} pixels'.format((delta > 8).sum()))
    print('delta value < -7        : {} pixels'.format((delta < -7).sum()))
    print('maximum delta value     : {} '.format(np.amax(delta)))
    print('minimum delta value     : {} '.format(np.amin(delta)))
    print('delta shape             : ' + str(delta.shape))

    delta = np.clip(delta, -7, 8)
    outnumpy = outnumpy + delta
    print('\nSuper resolution + 4 bit delta (PSNR): {:.2f}'.format(psnr(outnumpy, tarnumpy)))
