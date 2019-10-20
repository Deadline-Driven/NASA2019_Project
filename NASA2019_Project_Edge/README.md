# Requirement
You will need to create standalone android ndk, which can found some guide at [Android NDK]("https://developer.android.com/ndk")
This project we use android-ndk-r17b, and create standalone sdk by the following command
```
build/tools/make-standalone-toolchain.sh --arch=arm64 --platform=android-28 --install-dir=$HOME/Documents/android-ndk-toolchain
```

# Build Instruction
```
mkdir build
cmake ../
make
```

## Experiment
|-----|-----|
|Device| Oppo Reno Z|
|SoC | Mediatek Helio P90 |
|Overall Executing Time | ~350ms|
|On-device Executing Time| ~300ms|
