# 🌱 CropSky - AI-Powered Crop Disease Detection

**CropSky** is a pioneering tech startup revolutionizing the agriculture industry with AI, robotics, drones, and IoT. This project is a core component of our mission: a deep learning-based system for detecting plant diseases from leaf images.

## 📌 Overview
The goal of this project is to accurately detect plant diseases using Convolutional Neural Networks (CNNs). It utilizes the **New Plant Diseases Dataset** and implements a user-friendly web application using **Streamlit**. The system can identify 38 different classes of plant diseases and healthy leaves, providing instant diagnosis and expert recommendations for treatment.

## ✨ Features
- **Multi-Class Detection**: Capable of detecting 38 different plant diseases and healthy states across various species (Apple, Corn, Grape, Potato, Tomato, etc.).
- **User-Friendly Interface**: Built with Streamlit for easy image uploading and instant results.
- **Actionable Insights**: Provides specific cures and next steps for each detected disease.
- **Robust Model**: Custom CNN architecture optimized for accuracy and generalization.

## 📂 Project Structure
```
official cropsky/
├── app.py                                          # Streamlit web application entry point
├── crop_disease_detection_my_official_model.keras  # Trained CNN model file
├── model.ipynb                                     # Notebook for model training and architecture
├── testing.ipynb                                   # Notebook for model testing and validation
├── OBSERVATIONS.md                                 # Detailed notes on model variations and experiments
├── requirements.txt                                # Python dependencies
├── README.md                                       # Project documentation
├── LICENSE                                         # MIT License
└── images/                                         # Sample images for testing
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "official cropsky"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To start the web interface, run the following command:
```bash
streamlit run app.py
```
The application will open in your default web browser.

## 🧠 Model Architecture & Enhancements
Our model significantly improves upon standard reference implementations:

| Feature | Reference Model | CropSky Model | Benefit |
|---------|----------------|---------------|---------|
| **Layers** | 2 Convolutional Layers | **5 Convolutional Blocks (10 Layers)** | Extracts deeper, more complex patterns. |
| **Filters** | Max 64 | **Up to 512** | Learns richer feature representations. |
| **Padding** | Unspecified | **'Same' Padding** | Retains spatial dimensions and edge information. |
| **Regularization** | None | **Dropout (0.25, 0.4)** | Prevents overfitting for better generalization. |
| **Pooling** | 2 MaxPool Layers | **5 MaxPool Layers** | Reduces dimensionality while preserving features. |
| **Dense Layer** | 64 Units | **1500 Units** | Captures complex relationships for classification. |
| **Classes** | 3 Classes | **38 Classes** | Comprehensive disease coverage. |
| **Optimizer** | Adam (Default LR) | **Adam (LR=0.0001)** | Stable convergence and optimized weights. |

For more details on model variations and experiments, please refer to [OBSERVATIONS.md](OBSERVATIONS.md).

## 📊 Dataset
This project uses the **New Plant Diseases Dataset** (Augmented).
- **Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Author**: Vipul Upadhayay

*Note: If you use this dataset in research, please cite the original authors.*

## 🛠️ Technologies Used
- **TensorFlow/Keras**: Deep Learning framework.
- **OpenCV**: Image processing.
- **Streamlit**: Web application framework.
- **NumPy & Pandas**: Data manipulation.
- **Matplotlib**: Visualization.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by CropSky Team*