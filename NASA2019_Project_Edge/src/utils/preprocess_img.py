import cv2
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Convert img to format that can inference in NeuroPilot')
parser.add_argument('--input_image', type=str, required=True, help='input image to be convert')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output file')
opt = parser.parse_args()

cv2img = cv2.imread(opt.input_image)
cv2img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2YCR_CB)
(cv2_y, cv2_cr, cv2_cb) = cv2.split(cv2img)
print(cv2_cr.shape)
print(cv2_cb.shape)
cv2_cr.tofile("tmp_cv2_cr.bin")
cv2_cb.tofile("tmp_cv2_cb.bin")
cv2_y = cv2_y.astype(np.float32)
cv2_y /= 255.0
cv2_y = np.expand_dims(cv2_y, 0)
cv2_y = np.expand_dims(cv2_y, 0)

cv2_y.tofile(opt.output_filename)
