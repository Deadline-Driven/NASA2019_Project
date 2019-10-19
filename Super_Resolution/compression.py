import sys
import os
import numpy as np
import cv2

def main():
    train_dir = 'NASA_MSLMHL_0009_EXCERPT/train/images_o'
    val_dir = 'NASA_MSLMHL_0009_EXCERPT/test/images_o'

    train_com_dir = 'NASA_MSLMHL_0009_EXCERPT/train/images_s'
    val_com_dir = 'NASA_MSLMHL_0009_EXCERPT/test/images_s'

    scalefactor = 0.5

    train_file = [os.path.join(train_dir, filename) for filename in os.listdir(train_dir)]
    for st in train_file:
        res = os.path.join(train_com_dir, os.path.splitext(os.path.basename(st))[0]) + '_s.png'
        im = cv2.imread(st)
        im = cv2.resize(im, (0, 0), fx=scalefactor, fy=scalefactor, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(res, im)
        print(res + '  XXX  ' + st)

    val_file = [os.path.join(val_dir, filename) for filename in os.listdir(val_dir)]
    for st in val_file:
        res = os.path.join(val_com_dir, os.path.splitext(os.path.basename(st))[0]) + '_s.png'
        im = cv2.imread(st)
        im = cv2.resize(im, (0, 0), fx=scalefactor, fy=scalefactor, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(res, im)
        print(res + '  AAA  ' + st)

if __name__ == '__main__':
    sys.exit(main() or 0)
