# Face Mask Detection Using CNN
## Overview
A Convolutional Neural Network is a deep learning model designed for processing and analyzing visual data by using layers that automatically learn to detect features like edges, textures, and patterns. It consists of convolutional layers, pooling layers, and fully connected layers to classify and recognize objects in images. CNNs excel at tasks such as image classification, object detection, and facial recognition due to their ability to capture spatial hierarchies and patterns.
### CNN Model Pipeline
First, images are resized to a standardized resolution of 128×128 pixels and converted into the RGB color space to ensure consistent color representation. The preprocessed RGB images, with a resolution of 128×128 pixels, are then fed into a customized Convolutional Neural Network designed to extract relevant features for classifying whether a person is wearing a mask or not.

Necessary libraries, including **NumPy** for mathematical calculations and **TensorFlow** for building and training the CNN model, are imported. After preprocessing, images with masks are labeled as 1, and images without masks are labeled as 0. The dataset is then split into training and testing sets using the **train_test_split** function from the **sklearn** library, with 80% of the data allocated for training and 20% for testing.

The image arrays are combined into a single array, and each pixel value is normalized to be between 0 and 1 by dividing by 255. The **Customized CNN model** is constructed with 10 layers, including convolutional layers with 32, 64, and 128 filters, followed by max-pooling layers to down-sample the feature maps.

After the convolutional and pooling layers, a flattening layer converts the 2D feature maps into a 1D array. Dense layers with 128 and 64 neurons are applied, followed by dropout layers to prevent overfitting. The final dense layer has 2 neurons, corresponding to the binary classification task (masked or unmasked).

The model is compiled using the **Adam optimizer** and the **sparse categorical cross-entropy** loss function. The model is trained with a batch size=32 for epoch=20, with 10% of the training data reserved for validation using **validation_split=0.1**. After training, the model’s performance is evaluated on the test set using accuracy, precision, recall, and F1-score metrics.
