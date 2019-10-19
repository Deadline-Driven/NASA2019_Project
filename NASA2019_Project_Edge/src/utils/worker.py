import logging
from pathlib import Path
import numpy as np
import torch
from converters import pytorch2savedmodel, savedmodel2tflite

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    
    data_dir = Path.cwd().joinpath('../../res')
    onnx_model_path = data_dir.joinpath('nasa_srmodel_rePS.onnx')
    saved_model_dir = str(data_dir.joinpath('saved_model'))
    logger.info(f'\nConvert ONNX model to Keras and save as saved_model.pb.\n')
    pytorch2savedmodel(onnx_model_path, saved_model_dir)

    logger.info(f'\nConvert saved_model.pb to TFLite model.\n')
    tflite_model_path = str(data_dir.joinpath('nasa_srmodel.tflite'))
    tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False)

main()
