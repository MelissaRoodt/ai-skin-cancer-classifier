# Introduction

In this project, we used a CNN model on the ISIC 2024 Skin Cancer Detection dataset to accurately classify malignant and benign lesions. The report details each stage: data handling, model design, training, and evaluation. Additionally, we highlighted the model’s effectiveness in supporting accurate diagnoses by evaluating its performance on a testing set and a never-seen image separate from the original dataset.

# Dataset

Our model used the ISIC 2024 – Skin Cancer Detection dataset with 3D-TBP, specifically the Slice-3D Permissive dataset. This dataset contains skin lesion image crops extracted from 3D TBP with metadata entries (International Skin Imaging Collaboration, 2024). The metadata contains fields like `isic_id` and `malignant` value, where 0 is benign, and 1 is malignant as seen in Figure 1-1.

The first step was to manually extract malignant and benign images into separate evenly distributed classes. We ended up with 294 malignant images and 294 benign images, totaling 588 images. The goal is to make a binary classification model since there are only two classes: malignant (1) or benign (0).

**Table 1: Example of metadata**

| isic_id       | malignant |
|---------------|-----------|
| ISIC_0015670  | 0.0       |
| ISIC_0015845  | 0.0       |

# Data Loading

We loaded the ISIC 2024 - Skin Cancer Detection dataset using TensorFlow's `image_dataset_from_directory` method. This created a pipeline that allowed structured access to images stored in labeled folders (benign and malignant). We used the default values of the TensorFlow data pipeline, such as batches of 32 and image sizes of 256. This dataset was loaded in batches of 32 to ensure smooth training and validation, with each image in the dataset scaled to a consistent resolution of 256. To manage GPU memory and prevent Out of Memory (OOM) errors, GPU memory growth was enabled.

# Data Preprocessing

- **Scaling**: We scaled each image's pixel values to a range of 0-1 by dividing by 255. This normalized the data and improved model convergence.
- **Dataset Splitting**: We divided the dataset into training (70%), validation (20%), and testing (10%) sets to maintain a balanced evaluation of the model’s performance. This split was implemented using TensorFlow’s `take` and `skip` functions.

# Model Design

We modeled the architecture to follow a simple CNN layout with three convolutional layers, each followed by a max-pooling layer to reduce spatial dimensions and prevent overfitting. The structure contained the following layers:

- **Conv2D Layers**: We applied three layers with ReLU activations to capture spatial features and patterns in images.
- **MaxPooling2D Layers**: We reduced the feature map size after each convolution layer, lowering the computational load.
- **Flattening**: We flattened the pooled features into a 1D array for the dense layers.
- **Fully Connected Layers**: Finally, we added a dense layer with 256 units using ReLU activation to consolidate features, followed by a sigmoid-activated dense layer with 1 unit for binary classification.

The model was compiled with the Adam optimizer, BinaryCrossentropy loss, and metrics set to accuracy, which is ideal for a binary classification task.

# Model Training and Validation

We trained the model over 20 epochs, using the TensorBoard callback to log training and validation metrics. We visualized the metrics to ensure model performance over epochs, helping track loss and accuracy trends.

**Training Observations**: Loss and accuracy plots showed a consistent improvement over epochs as seen in Figures 1 and 2.

![Alt text](/Resources/loss.jpg?raw=true "Dashboard")
- **Figure 1**: Model Loss
![Alt text](/Resources/acc.jpg?raw=true "Dashboard")
- **Figure 2**: Model Accuracy

# Model Evaluation and Testing

For model evaluation, we calculated:

- **Precision, Recall, and Accuracy**: Using TensorFlow’s metrics on the test dataset, the model achieved a good performance score for binary classification.

**Results**:

- **Precision**: 0.88 (demonstrating the model’s ability to correctly identify malignant cases).
- **Recall**: 0.85 (showing effective sensitivity towards malignant cases).
- **Accuracy**: 0.87 (indicating overall prediction performance across benign and malignant classes).

The model's effectiveness was further verified by testing it on an unseen external image, where it successfully predicted either benign or malignant, enhancing model confidence for real-world deployment.

# Conclusion

This model demonstrates strong capability in classifying skin cancer images, as seen in the testing results. Future improvements could involve model tuning, advanced data augmentation, and cross-validation for better performance.

# References

International Skin Imaging Collaboration. 2024. SLICE-3D 2024 Permissive Challenge Dataset. International Skin Imaging Collaboration [https://doi.org/10.34970/2024-slice3d-permissive](https://doi.org/10.34970/2024-slice3d-permissive) Date of access: 11 Nov. 2024.
