# Pix2Pix Implementation from Scratch (PyTorch)

This project implements the **Pix2Pix image-to-image translation model** from scratch using **PyTorch**.  
The model is trained on the [Pix2Pix Maps Dataset](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset), where the task is to translate **satellite images into map views** and vice versa.

---

## 📌 Project Overview
Pix2Pix is a **conditional GAN (Generative Adversarial Network)** designed for paired image-to-image translation tasks.  
- **Generator**: U-Net-based architecture to generate translated images.  
- **Discriminator**: PatchGAN discriminator to classify real vs. fake image patches.  
- **Loss Functions**: Combination of adversarial loss and L1 reconstruction loss.  

---

## 🚀 Features
- Built **from scratch** using PyTorch (no external GAN libraries).  
- Supports training and testing on custom datasets.  
- Visualizes generated results during training.  
- Easily extendable to other paired datasets beyond maps.  

---

## 📂 Dataset
The dataset used is the **Maps Dataset** from Kaggle:  
👉 [Pix2Pix Maps Dataset](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset)

It contains pairs of images where:  
- **Input**: Satellite photo.  
- **Target**: Map view.  
