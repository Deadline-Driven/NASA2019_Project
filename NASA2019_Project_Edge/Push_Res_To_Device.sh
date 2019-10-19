adb wait-for-device

adb push res/nasa_srmodel.tflite /data/local/tmp
adb push res/target.bin.npy /data/local/tmp/target.bin
adb push build/SR_visionrecovery /data/local/tmp
