#!/bin/bash

INPUT_FILE=$1
if [ -z "$INPUT_FILE" ];
then
	echo "Usage: ./execute_process.sh <Input image path>"
else
	echo "Input File: "
	echo $INPUT_FILE

	cd src/utils
	python preprocess_img.py --input_image $INPUT_FILE --output_filename target.bin
	mv target.bin ../../res
	cd ../../

	adb wait-for-device

	adb push res/nasa_srmodel.tflite /data/local/tmp
	adb push res/target.bin /data/local/tmp/target.bin
	adb push build/SR_visionrecovery /data/local/tmp
	adb shell /data/local/tmp/SR_visionrecovery
	adb pull /data/local/tmp/result.bin res/result.bin

	cd src/utils
	python postprocess_img.py --input_data "../../res/result.bin" --output_file "recover.png"
fi

