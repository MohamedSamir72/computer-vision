# 🧠 Variational Autoencoder (VAE)

A PyTorch implementation of a **Variational Autoencoder (VAE)** for image reconstruction and latent space learning.  
This project is part of the [Computer Vision](https://github.com/MohamedSamir72/computer-vision) repository.
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

---

## Features
- Encoder–Decoder architecture built with PyTorch  
- Reconstruction and KL Divergence loss combination  
- Training and evaluation loops with automatic checkpoint saving  
- Support for CPU and GPU (CUDA)  
- Visualization of reconstructed images and latent space  

---

## 📂 Project Structure
```
variational_Autoencoder/
│
├── model.py # Defines VAE architecture
├── train.py # Training loop and loss computation
├── utils.py # Helper functions (plotting, saving, etc.)
├── dataset.py # Dataset loading and preprocessing
├── results/ # Saved models, losses, and reconstructions
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

---

## 🧩 Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```
## 🧠 Model Overview

#### The VAE consists of:

Encoder: compresses the input image into a latent representation
Latent Space: parameterized by mean μ and log-variance log(σ²)
Decoder: reconstructs the original image from the latent vector

## Loss Function

The total loss function combines Reconstruction Loss and Kullback–Leibler (KL) Divergence:
```
L = α × Reconstruction Loss + β × KL Divergence
```

| Symbol        | Name                       | Description                                                                 | Typical Value |
| :------------ | :------------------------- | :-------------------------------------------------------------------------- | :------------ |
| **α (alpha)** | Reconstruction Loss Weight | Controls the importance of pixel reconstruction accuracy                    | `1.0`         |
| **β (beta)**  | KL Divergence Weight       | Controls the strength of latent regularization (smoothness of latent space) | `1.0` |

```
RECONSTRUCTION_LOSS_WEIGHT: float = 1.0
KLD_LOSS_WEIGHT: float = 1.0
```