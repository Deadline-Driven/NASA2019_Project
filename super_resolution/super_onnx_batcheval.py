from __future__ import print_function
import argparse
from time import time
from math import log10, sqrt
import numpy as np
import onnxruntime
import cv2
from os import listdir
from os.path import join, splitext, basename

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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_dir', type=str, required=True, help='input images folder to be inferenced')
parser.add_argument('--target_dir', type=str, required=True, help='target images folder')
parser.add_argument('--model', type=str, required=True, help='onnx model file to be used')
parser.add_argument('-u', '--upscale_factor', help='set upscale factor', default=2, type=int)
opt = parser.parse_args()

testdata_pools = [join(opt.input_dir, x) for x in listdir(opt.input_dir) if is_image_file(x)]
total = len(testdata_pools)
total = float(total)
sum_nn = 0
sum_bili = 0
sum_area = 0
sum_cubic = 0
sum_lanc = 0
sum_sr = 0
sum_deltasr = 0



for test_image in testdata_pools:
    target_image = join(opt.target_dir, basename(test_image).replace('_s.png', '.png'))
    
    cv2img = cv2.imread(test_image)
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
    cv2_out_img = cv2.cvtColor(cv2_out_img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(opt.output_filename, cv2_out_img)


    #print('Output image saved to', opt.output_filename)
    #print(' ')


    cv2_target = cv2.cvtColor(cv2.imread(target_image), cv2.COLOR_BGR2RGB)
    print('PSNR evaluation:')


    cv2_test_img = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)

    nn_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
    now_nn = psnr(nn_img, cv2_target)
    print('resize with OpenCV INTER_NEAREST : {:.2f}'.format(now_nn))

    bili_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
    now_bili = psnr(bili_img, cv2_target)
    print('resize with OpenCV INTER_LINEAR  : {:.2f}'.format(now_bili))

    area_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_AREA)
    now_area = psnr(area_img, cv2_target)
    print('resize with OpenCV INTER_AREA    : {:.2f}'.format(now_area))

    bicu_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    now_cubic = psnr(bicu_img, cv2_target)
    print('resize with OpenCV INTER_CUBIC   : {:.2f}'.format(now_cubic))

    lanc_img = cv2.resize(cv2_test_img, (0, 0), fx=2, fy=2, interpolation = cv2.INTER_LANCZOS4)
    now_lanc = psnr(lanc_img, cv2_target)
    print('resize with OpenCV INTER_LANCZOS4: {:.2f}'.format(now_lanc))

    #cv2_pyt_out_img = cv2.cvtColor(cv2.imread(opt.output_filename), cv2.COLOR_BGR2RGB)
    now_sr = psnr(cv2_out_img, cv2_target)
    print('super resolution OpenCV (python) : {:.2f}'.format(now_sr))

    #cv2_cpp_out_img = cv2.cvtColor(cv2.imread('output_cpp.png'), cv2.COLOR_BGR2RGB)
    #print('super resolution OpenCV (c++)    : {:.2f}'.format(psnr(cv2_cpp_out_img, cv2_target)))

    print('\nDelta evaluation (4 bits):')
    outnumpy = np.array(cv2_out_img).astype('int16') #image generated from c++ inference program
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
    now_deltasr = psnr(outnumpy, tarnumpy)
    print('\nSuper resolution + 4 bit delta (PSNR): {:.2f}\n'.format(now_deltasr))
    
    sum_nn = sum_nn + now_nn
    sum_bili = sum_bili + now_bili
    sum_area = sum_area + now_area
    sum_cubic = sum_cubic + now_cubic
    sum_lanc = sum_lanc + now_lanc
    sum_sr = sum_sr + now_sr
    sum_deltasr = sum_deltasr + now_deltasr
    
    print('avgNN={:.2f}, avgBili={:.2f}, avgArea={:.2f}, avgCubic={:.2f}, avgLanc={:.2f},\n avgSr={:.2f}, avgDeltasr={:.2f}'.format(sum_nn/total, sum_bili/total, sum_area/total, sum_cubic/total, sum_lanc/total, sum_sr/total, sum_deltasr/total))
