#   Document
# Video Demonstration


https://github.com/magnus-6/projects_dis/assets/121368258/641a1ea8-8652-4364-aeeb-1ac692528c52


## Introduction
This project focuses on creating a high-performance anti-spoofing detection system using a Convolutional Neural Network (CNN) model. It includes a Python-based graphical user interface (GUI) and a Raspberry Pi 5 implementation to ensure practical application.

## Goals and Workflow
### Goals
- Develop a robust anti-spoofing detection system.
- Achieve high accuracy with a pre-trained Support Vector Machine (SVM) model.
- Implement a Python-based GUI.
- Ensure software-to-hardware implementation on a Raspberry Pi 5.

### Workflow
1. **Data Collection**: Images from 233 individuals, each with 12 genuine and 12 fake images.
2. **Data Preprocessing**: Using MTCNN for face and facial landmark detection.
3. **Feature Extraction**: Utilizing models like FaceNet, LBP, HOG, VGG16, and VGG19.
4. **Training and Testing**: SVM training and testing, with DenseNet for improved performance.
5. **Performance Evaluation**: Accuracy assessment, including real-time testing.
6. **Hardware Implementation**: Setup and testing on Raspberry Pi.
7. **GUI Implementation**: Development and integration using Python and Tkinter.

## Dataset Description

### Antispoofing Dataset
- **Total Folders**: 233
- **Each Folder**: 12 original and 12 fake images

### Facial Recognition Dataset
- **Total Folders**: 233
- **Each Folder**: 12 images

**The dataset is created and organized by me by manually collecting images from participants. This dataset is completely new and unique**, with separate sets for processed and non-processed images. The total size of the dataset is calculated as follows:

- **Antispoofing Dataset**: 2 * (233 * 24) = 11,184 images
- **Facial Recognition Dataset**: 2 * (233 * 12) = 5,592 images

Therefore, the overall dataset size is 16,776 images.

## Data Preprocessing
- **Resizing**: Images to 224x224 pixels.
- **Normalization**: Pixel values to the [0, 1] range.

## Feature Extraction Models
Feature extraction using FaceNet, LBP, HOG, VGG16, and VGG19, with details on image size, feature vector dimensions, color mode, and processing time.

## Training and Testing Protocol
- **Folder Creation**: Separate training and testing folders.
- **Image Selection**: Equal distribution of original and fake images for training and testing.
- **SVM Training and Testing**: Using original and fake images.
- **Transfer Learning**: Using DenseNet for enhanced performance.

## Performance and Evaluation
### Antispoofing - Uncropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 233             | 74.356  | 91.309 | 97.818| 95.708 | 96.352 |

### Antispoofing - Cropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 233             | 60.30   | 91.881 | 87.232| 92.239 | 65.379 |

### Facial Recognition - Uncropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 233             | 41.732  | 64.306 | 78.239| 95.422 | 93.772 |

### Facial Recognition - Cropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 233             | 90.272  | 14.235 | 20.974| 20.815 | 8.948  |

### DenseNet Performance

![Screenshot 2024-06-17 225130](https://github.com/magnus-6/projects_dis/assets/121368258/2d810ad8-76bd-48b5-a953-1ed73ab2b9b8)
## Remark

### DenseNet Performance
The DenseNet model achieved an **AUC of 0.999775** and an **accuracy of 0.955** for anti-spoofing detection.

### Facial Recognition - Cropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 233             | **90.272**  | 14.235 | 20.974| 20.815 | 8.948  |

Given that **FaceNet performs best for facial recognition**, achieving an accuracy of **90.272%** on cropped data, it will be used for the recognition system after preprocessing using MTCNN. Therefore, **DenseNet will be used for anti-spoofing**, and **FaceNet will be utilized for recognition** in this system.

### Dataset Composition for DenseNet
- **Training and Testing Set**: 200 images total (100 training, 100 testing)
  - **Training**: 50 original, 50 fake
  - **Testing**: 50 original, 50 fake

### Dataset Composition for Validation
- **Validation Set**: 400 images (200 original, 200 fake)
  - **Original Images**: Captured using a laptop camera.
  - **Fake Images**: Displayed on a mobile phone screen and captured with a laptop camera.

## Hardware Implementation
- **Implementation on Raspberry Pi**: Included hardware and software setup, conducting training and testing, and overcoming implementation challenges.

## GUI Implementation
- **Development of GUI**: Using Python and Tkinter.
- **Integration**: GUI with the system for practical applications.

## Conclusion
This project successfully developed a robust anti-spoofing detection system with high accuracy, leveraging advanced machine learning techniques and practical hardware implementation. The Python-based GUI and Raspberry Pi integration further enhance the system's usability and applicability in real-world scenarios.


