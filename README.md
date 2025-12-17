# 🍌 Sweet Spot: Banana Ripeness Classification System

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio)

**Sweet Spot** is a machine learning-powered web application designed to classify the ripeness level of bananas. Developed as a generic study for *Modelling and Simulation*, this system compares the performance of custom CNN architectures against industry-standard Transfer Learning models.

## 🎯 Project Overview

This application classifies banana images into four distinct categories:
1.  **Unripe** (Green, firm)
2.  **Ripe** (Yellow, optimal)
3.  **Overripe** (Brown spots)
4.  **Rotten** (Black, spoiled)

The system provides a user-friendly interface where users can upload images and select between three different neural network architectures to see how they analyze the same image differently.

## ✨ Features

* **Multi-Model Support:** Switch instantly between **VGG19**, **ResNet50**, and the custom **GiMaTag CNN**.
* **Real-time Analysis:** Get immediate confidence scores for each ripeness class.
* **Interactive Dashboard:** A clean, 2-column UI built with Gradio for easy image uploading and result visualization.
* **TorchScript Deployment:** Uses optimized `.pt` models for faster inference and lightweight deployment.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher (Developed on Python 3.13)
* Git (optional, for cloning)

### 1. Clone the Repository
```bash
git clone https://github.com/WakenMac/SweetSpot-CNN-Banana-Ripeness-Classification-Model-MaS-LE.git
cd SweetSpot-CNN-Banana-Ripeness-Classification-Model-MaS-LE
```
### 2. Install Dependencies
It is recommended to use a virtual environment.
```
pip install -r requirements.txt
```

If requirements.txt is missing, install the core libraries manually:
```
pip install torch torchvision gradio pillow numpy
```

### 3. Setup Models
Ensure the following TorchScript model files are present in the models/ directory:
* gimatag_complete.pt
* resnet50_complete.pt
* vgg19_complete.pt

(If you only have .pth weights, run the provided conversion scripts save_transfer_models.py and save_gimatag.py to generate these files.)

### 4. Run the Application
```
python3 app.py
```

Open your browser to the local URL provided (usually http://127.0.0.1:7860).

### 🧠 Model Architectures

| Model | Type | Description |
| :--- | :--- | :--- |
| **GiMaTag CNN** | Custom | A lightweight, 3-block Convolutional Neural Network designed specifically for this dataset. |
| **VGG19** | Transfer Learning | A deep CNN using pre-trained ImageNet weights, fine-tuned with a custom classifier head. |
| **ResNet50** | Transfer Learning | A 50-layer Residual Network using skip connections to learn deep features, adapted for 4-class classification. |

### 📂 Project Structure

```text
Banana-Classification-App/
├── models/                  # Trained TorchScript models
│   ├── gimatag_complete.pt
│   ├── resnet50_complete.pt
│   ├── vgg19_complete.pt
│   └── README.md
├── app.py                   # Main Gradio application
├── save_gimatag.py          # Script to convert GiMaTag weights to .pt
├── save_transfer_models.py  # Script to convert Transfer models to .pt
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

### 📊 Datasets

The model was trained using a combination of primary data collected by the researchers and publicly available datasets for augmentation and validation:

1.  **GiMaTag Dataset** (Primary)
    * Original dataset collected and annotated by the researchers (**Gi**gawin, **Ma**clang, **Tag**le).
    * Contains images of local banana varieties captured in controlled and real-world environments.

2.  **Fayoum University Banana Dataset**
    * External dataset used for training augmentation and benchmarking.
    * *Source: Biswas et al., Fayoum University.*

3.  **Banana Image Dataset (Shariar)**
    * Supplementary dataset used to increase class diversity.
    * *Source: Shariar (Kaggle/Mendeley Data).*

*All images were pre-processed and normalized to 224x224 dimensions to ensure consistency across architectures.*

### 👥 Researchers & Credits
Developed in completion of the Modelling and Simulation course at the University of Southeastern Philippines (USeP), College of Information and Computing.

* Dave Shanna Marie E. Gigawin
* Waken Cean C. Maclang
* Allan C. Tagle

Note: This is an academic project. The models are trained on specific datasets and may vary in performance on real-world images with different lighting conditions.
