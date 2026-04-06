<div align="center">
<h1 align="center"><strong>🛣 Land-Cover-Semantic-Segmentation-PyTorch:<h6 align="center">An end-to-end Image Segmentation project</h6></strong></h1>

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
[![Generic badge](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE) 
</div>

----

## 📚 Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup without Docker (OS Independent)](#setup-without-docker)
  - [Setup and Running with Docker](#with-docker)
  - [Running the project](#running-the-project)
- [License](#license)

----

## 📌 Overview <a name="overview"></a>
An end-to-end Computer Vision project focused on Image Segmentation (specifically Semantic Segmentation). Originally built with the LandCover.ai dataset, the project provides a flexible template that can be applied to train models on any semantic segmentation dataset. 

It leverages modern architectures (UNet, DeepLabV3+, Transformer-based models, etc.) via `segmentation-models-pytorch` and `timm`, providing deep customizability through configuration files. You can train on full multiclass datasets and prompt the model at inference time to predict specific selective classes of interest (e.g., passing `test_classes = ['parking']` to extract only the parking zones).

----

## 📂 Directory Structure <a name="directory-structure"></a>

```text
LandCover/
├── assets/                     # Images and static assets
├── config/                     # Model and training configurations
├── data/                       # Contains dataset (e.g., train/ and test/)
├── notebooks/                  # Jupyter notebooks for data exploration
├── src/                        # Main source code logic
│   ├── utils/                  # Helper modules and functions
│   ├── compare_models.py       # Compare multiple models performance
│   ├── inference.py            # Run inference on new images
│   ├── test_models.py          # Test specific models framework
│   └── train_model.py          # Main model training script
├── activate_env.sh             # Activates virtual environment and shows commands
├── Dockerfile                  # Docker setup for the project
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

----

## 💫 Features & Demo <a name="demo"></a>
- **State-of-the-art Archs**: Easily switch between robust classic (U-Net, DeepLabV3) and newer Transformer models.
- **Cross-Platform**: Configured to run on Windows, Linux, and macOS without strict OS-dependent friction.
- **AutoML Configuration**: Customize optimizers, learning rate, and architectures in standard YAML files.

---

## 🚀 Getting Started <a name="getting-started"></a>

### ✅ Prerequisites <a name="prerequisites"></a>
 
- **Dataset Needs**:
Download your dataset (e.g., from [LandCover.ai](https://landcover.ai.linuxpolska.com/)) and arrange the imagery such that the `images` and `masks` directories are placed under the `data/train` and `data/test` directories inside this project.

### 💻 Setup without Docker (OS Independent) <a name="setup-without-docker"></a>
 
We strongly recommend using an isolated virtual environment (`conda` or Python's native `venv`). This setup is platform-independent.

1. **Clone the repository:**
```shell
# Assuming you cloned or downloaded the project
cd Land-Cover-Semantic-Segmentation-PyTorch
```

2. **Create a virtual environment:**

- **Using `venv` (Windows):**
```shell
python -m venv segment_env
segment_env\Scripts\activate
```

- **Using `venv` (Linux/macOS):**
```shell
python3 -m venv segment_env
source segment_env/bin/activate
```

- **Using `conda`:**
```shell
conda create --name segment_env python=3.10
conda activate segment_env
```

3. **Install PyTorch:**
PyTorch distribution varies heavily depending on whether you plan to use an NVIDIA GPU or just compute on your CPU. To ensure proper Hardware/CUDA utilization during training:

*Example for NVIDIA GPU (CUDA 12.1/12.4):*
```shell
pip install torch torchvision torchaudio
```
*(By default, standard pip installation on PyPI retrieves the fully loaded CUDA setup! Run this to use your RTX/GTX GPU.)*

*Example for CPU-only (if you DO NOT have a GPU):*
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. **Install Remaining Packages:**
```shell
pip install -r requirements.txt
```

### 🐳 Setup and Running with Docker <a name="with-docker"></a>
 
This project comes with a CUDA-ready `Dockerfile` to seamlessly support GPU computation out-of-the-box in isolated environments.
 
1. **Build the image:**
```shell
docker build -t segment_project_image .
```

2. **Run the container:**
By default, the `Dockerfile` runs the inference/test script. To enable CUDA (GPUs), add the `--gpus all` flag if you have NVIDIA Container Toolkit set up.
```shell
docker run --gpus all --name segment_container -d segment_project_image
```

3. **Retrieve outputs:**
Once finished running, you can pull your trained models or test outputs onto your local machine:
```shell
docker cp segment_container:/segment_project/models ./models
docker cp segment_container:/segment_project/output ./output
```

4. **Clean up:**
```shell
docker stop segment_container
docker rm segment_container
```

### 🤖 Running the project <a name="running-the-project"></a>
 
From your project's root directory, you can utilize the scripts located in the `src/` folder.

1. **Training the model:**
```shell
python src/train_model.py --model unet
```
2. **Testing the model (with images and ground-truth masks):**
```shell
python src/test_models.py --model unet
```
3. **Inference (predicting on new images without masks):**
```shell
python src/inference.py
```
4. **Compare Models:**
```shell
python src/compare_models.py --models unet deeplabv3 linknet
```

----

## 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](LICENSE).

<p align="right">
 <a href="#top"><b>🔝 Return </b></a>
</p>
