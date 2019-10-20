# NASA2019_Project   
The NeuroPilot data recovery solution project   
https://2019.spaceappschallenge.org/challenges/planets-near-and-far/raiders-lost-data/teams/deadline-driven/project
   
   
```
NASA2019_Project
├── NASA2019_Project_Edge
│   │
│   ├── CMakeLists.txt // Build cmake file
│   │
│   ├── build
│   │   ├── SR_visionrecovery
│   │   └── RNN_dataimputation
│   ├── res
│   │   ├── nasa_srmodel.tflite // Super Resolution model file
│   │   └── nasa_rnnmodel.tflite // RNN model file
│   └── src
│       ├── TFLiteSR.cpp // Super Resolution inference for NeuroPilot
│       ├── TFLiteRNN.cpp // RNN inference for NeuroPilot
│       └── utils
│           ├── preprocess_img.py // Convert image in to pre-process data
│           ├── postprocess_img.py // Recover data back to image
│           └── converters.py // Convert ONNX to TFLite  
└── Super_Resolution
    ├── main.py                 // Model training script
    ├── model.py                // Model architecture
    ├── dataset.py              // Data preprocessing
    ├── data.py                 // Data preprocessing
    ├── compression.py          // Perform downsampling
    ├── super_resolve_rePS.py   // Perform inference and export ONNX model with PIL and PyTorch model
    ├── super_onnx_rePS.py      // Perform inference and output recovery image with OpenCV and ONNX model
    ├── super_onnx_batcheval.py // Evaluation on the test dataset
    ├── psnr_function.py        // Evaluation algorithm
    ├── nasa_srmodel_rePS.onnx  // ONNX model without pixel shuffle layer
    └── model_epoch_200.pth     // PyTorch model without pixel shuffle layer 
```
   
### Image Denoise Performance     
   
| Method        | Average PSNR           |
| ------------- |:-------------:|
| INTER_NEAREST             |   29.48  |
| INTER_LINEAR              |  29.92   |
| INTER_AREA                |   29.48  |
| INTER_CUBIC               | 31.28    |
| INTER_LANCZOS4            |   31.49  |
| Our Super Resolution          |   32.78  |   
   
NeuroPilot inference time: 299.53 ms   
   
### Training Performance  
Dataset: Food-11   
Model  : ResNet18  
  
| Dataset  |  Top 1 Accuracy | Top 3 Accuracy |
| ------------- |:-------------:|:-------------:|
| Original data|  2346/3347 (70.09%)   |  2945/3347 (87.99%)|
| Our method |  2302/3347 (68.78%)   | 2959/3347 (88.41%)|
   
### Demonstration
![](https://i.imgur.com/L2xikBG.png)

### Missing Data Imputation
![](https://i.imgur.com/k1v5yLG.png)

### MSE Loss Figure(Train & Test)
![](https://i.imgur.com/b5ao765.png)
