# NASA2019_Project   
The NeuroPilot data recovery solution project   
https://2019.spaceappschallenge.org/challenges/planets-near-and-far/raiders-lost-data/teams/deadline-driven/project
   
NASA2019_Project
├── NASA2019_Project_Edge
│   ├── build
│   │   ├── 
│   │   └── 
│   ├── res
│   │   ├── 
│   │   └── 
│   └── src
│       ├── 
│       └── 
└── Super_Resolution
    ├── main.py
    ├── model.py
    ├── dataset.py
    ├── compression.py
    ├── super_resolve_rePS.py
    ├── super_onnx_rePS.py
    ├── super_onnx_batcheval.py
    ├── psnr_function.py
    ├── nasa_srmodel_rePS.onnx
    └── model_epoch_200.pth
   
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

### Missing Data Imputation & MSE Loss Figure(Train & Test)
![](https://i.imgur.com/0cZK5qV.png)
