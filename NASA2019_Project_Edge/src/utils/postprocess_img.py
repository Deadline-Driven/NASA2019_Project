import cv2
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Convert data back to image after inference by NeuroPilot')
parser.add_argument('--input_data', type=str, required=True, help='input data to be convert')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
opt = parser.parse_args()

input_data = np.fromfile(opt.input_data,  dtype=np.float32).reshape(1,4,72,96)
print(input_data.shape)
img_out_y = np.zeros((1, 1, 144, 192))

for i in range(144):
    for j in range(192):
        if i%2==0 and j%2==0:
            img_out_y[0][0][i][j] = input_data[0][0][i//2][j//2]
        if i%2==0 and j%2==1:
            img_out_y[0][0][i][j] = input_data[0][1][i//2][(j-1)//2]
        if i%2==1 and j%2==0:
            img_out_y[0][0][i][j] = input_data[0][2][(i-1)//2][j//2]
        if i%2==1 and j%2==1:
            img_out_y[0][0][i][j] = input_data[0][3][(i-1)//2][(j-1)//2]


cv2_cb = np.fromfile("tmp_cv2_cb.bin", dtype=np.uint8).reshape(72,96)
cv2_cr = np.fromfile("tmp_cv2_cr.bin", dtype=np.uint8).reshape(72,96)


cv2_out_img_y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
cv2_out_img_cb = cv2.resize(cv2_cb, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img_cr = cv2.resize(cv2_cr, (cv2_out_img_y.shape[1], cv2_out_img_y.shape[0]), interpolation = cv2.INTER_CUBIC)
cv2_out_img =  cv2.merge([cv2_out_img_y, cv2_out_img_cr, cv2_out_img_cb])
cv2_out_img = cv2.cvtColor(cv2_out_img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(opt.output_filename, cv2_out_img)

