# Plant_Disease_Prediction_CNN_Image_Classifirer
A convolutional neural network that classifies plant leaf images into 38 categories spanning 14 plant species, identifying healthy leaves versus 24 distinct diseases. Built with TensorFlow/Keras and trained on the PlantVillage dataset.

### Overview

This project trains an image classifier to recognize plant diseases from leaf photos. The goal is to support early disease detection, which can help farmers and gardeners intervene before a disease spreads.

### Dataset

**Source:** PlantVillage Dataset (Kaggle)
**Classes:** 38 (14 plant species, each healthy or affected by one of several diseases)

**Images used:** ~54,000 RGB images from the color subset of the dataset (the dataset also includes grayscale and segmented versions, not used here)

**Split:** 80% training / 20% validation, via ImageDataGenerator

### Model Architecture

A simple CNN built with tf.keras.Sequential:

|Layer| | Details|
|Conv2D|32 filters, 3x3, ReLU|
|MaxPooling2D| 2x2|
|Conv2D|64 filters, 3x3, ReLU|
|MaxPooling2D| 2x2|
|Flatten—Dense |256 units, ReLU|
|Dense (output)|38 units, Softmax|


**Input size:** 224 x 224 x 3
**Optimizer:** Adam
**Loss:** Categorical cross-entropy
**Epochs:** 5
**Batch size:** 32

### Results

**After 5 epochs:**

Metric Value
Training accuracy ~98.2%
Validation accuracy ~88.4%

*Known Limitation:* Overfitting

There's a noticeable gap between training accuracy (~98%) and validation accuracy (~88%), which indicates the model is overfitting to the training data rather than learning fully generalizable features. This is a known issue with the current version and is the next thing planned to be addressed, likely through some combination of:


*Data augmentation* (random rotation, flips, zoom, brightness shifts) to expose the model to more variation during training

*Dropout layers* between the dense layers to reduce reliance on specific neurons

*Batch normalization* to stabilize training

*Early stopping* based on validation loss, since validation loss starts climbing again after epoch 3 while training loss keeps falling

*L2 regularization* on the dense layers

*Reducing model capacity* or simplifying the architecture if augmentation/regularization alone aren't enough
