import cv2
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Convert data back to image after inference by NeuroPilot')
parser.add_argument('--input_data', type=str, required=True, help='input data to be convert')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
opt = parser.parse_args()

img_out_y = np.load(opt.input_data)
cv2_cb = np.load("tmp_cv2_cb.npy")
cv2_cr = np.load("tmp_cv2_cr.npy")


cv2_out_img_y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
cv2_out_img_cb = cv2.resize(cv2_cb, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img_cr = cv2.resize(cv2_cr, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img =  cv2.merge([cv2_out_img_y, cv2_out_img_cr, cv2_out_img_cb])
cv2_out_img = cv2.cvtColor(cv2_out_img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(opt.output_filename, cv2_out_img)

