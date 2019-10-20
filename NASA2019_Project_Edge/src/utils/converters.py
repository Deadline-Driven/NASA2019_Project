# modified from https://github.com/lain-m21/pytorch-to-tflite-example/blob/master/converters.py
from __future__ import absolute_import, division, print_function, unicode_literals

import shutil
import os
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.models import model_from_json
from onnx2keras import onnx_to_keras
import cProfile

def onnx2kerasmodel(onnx_model_path, keras_model_dir):
    onnx_model = onnx.load(onnx_model_path)
    input_names = ['input']
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names, 
                            change_ordering=True, verbose=False)
    model_json = k_model.to_json()
    with open(keras_model_dir + "/model.json", 'w') as json_file:
        json_file.write(model_json)
    k_model.save_weights(keras_model_dir + "/model.h5")
    print("Model Saved!")
    
def keras2tflite(keras_model_dir, tflite_model_path, quantize=False):
    network_define = keras_model_dir.joinpath("model_config.json")
    network_weight = keras_model_dir.joinpath("weights.h5")
    json_file = open(network_define, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(network_weight)
    print("Model Recovered!")
    converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print("tflite_model write at %s" % (tflite_model_path) )
    return tflite_model


def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    onnx_model = onnx.load(onnx_model_path)

    input_names = ['image_array']
    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=['input'],
                            change_ordering=False, verbose=True)

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        tf.saved_model.simple_save(
            sess,
            str(saved_model_dir.joinpath('1')),
            inputs={'image_array': k_model.input},
            outputs=dict((output.name, tensor) for output, tensor in zip(onnx_model.graph.output, k_model.outputs))
        )


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False):
    saved_model_dir = str(Path(saved_model_dir).joinpath('1'))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print("tflite_model write at %s" % (tflite_model_path) )
    return tflite_model

