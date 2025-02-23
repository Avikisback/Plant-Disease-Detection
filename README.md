🌱 Crop Disease Detection using CNNs

**Overview**:
The aim of this project is to detect plant diseases using Convolutional Neural Networks (CNNs).It uses the 'New Plant Diseases Dataset' from Kaggle and implements 'real-time detection' using OpenCV and TensorFlow.The web application for this project is built with 'Streamlit'.

**Features**:
1) Detects 38 different plant diseases using deep learning (CNN).
2) Users can upload images of plant leaves.
3) Provides suggested treatments for detected diseases.
4) Built with Streamlit for easier access.

**Dataset**:
The project utilizes the New Plant Diseases Dataset from Kaggle, consisting of images of diseased and healthy plant leaves across multiple species.

**Dataset Citation**:
Source-https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
Dataset Author: **Vipul Upadhayay**  

If you use this dataset in research or publication, consider citing the original dataset authors as per Kaggle's guidelines.

**Model Enhancements**:
1) The model consists of 10 convulation layers compared to the reference video containing only 2 convulational layers.More convulational layers allow the model to extract more complex patterns and hierarchical features from images.
2) The model starts with 32 filters and increases up to 512 filters, while in the reference video it goes to up 64 filters only.More filters allow the network to learn richer and more detailed feature representations.
3) The model includes padding in several convolution layers (padding='same'), which helps retain spatial dimensions, helps the corner pixels of the image to create impact as much as the central pixels, hence preventing excessive information loss while in the reference video padding is not specified.
4) The model includes Dropout(0.25,0.4), which helps prevent overfitting by randomly deactivating some neurons during training, while the reference video does not have dropout making it more prone to overfitting.
5) The model consists of 5 MaxPooling layers, while the reference video only has 2 MaxPooling layers.More MaxPooling layers help in reducing dimensionality while preserving important features, leading to better generalization.
6) The model has a Dense(1500) layer, while the reference video has only Dense(64) before the output layer. A larger dense layer allows better classification by leanring more complex relationships.
7) The model can detect up to 38 different output classes, while the reference video can only detect up to 3 output classes.
8) The model consists of learning_rate=0.0001, which helps with stable convergence and prevents overshooting the optimal weights, while the reference video uses the deafault Adam learning rate (0.001),which maybe too high and lead to instability.

Code Structure:
1) app.py - Streamlit-based UI for disease detection.
2) model.ipynb - CNN model implementation.
3) crop_disease_detection_my_official_model.keras- CNN model.
4) testing.ipynb - Testing of model with the testing data provided in the dataset.

Example Usage:
1) Upload an image of a plant leaf.
2) Click the Predict button.
3) The model predicts the disease and suggests remedies.

Next Steps:
1) Improve the model by fine-tuning hyperparameters.
2) Expand the dataset for better generalization.
3) Deploy the model on a dedicated cloud service.

Please read the OBSERVATIONS.md file for the details about the other model implementations.