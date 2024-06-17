# README Document

## Introduction
This project aims to develop a robust anti-spoofing detection system that can accurately distinguish between genuine and spoofed images. The primary goal is to create a software solution that maintains high performance across various conditions by leveraging a state-of-the-art Convolutional Neural Network (CNN) model to enhance accuracy. Additionally, the project implements a Python-based graphical user interface (GUI) and a software-to-hardware implementation on a Raspberry Pi 5 to ensure applicability in real-world scenarios.

## Table of Contents
1. [Goals and Workflow](#goals-and-workflow)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction Models](#feature-extraction-models)
5. [Training and Testing Protocol](#training-and-testing-protocol)
6. [Performance and Evaluation](#performance-and-evaluation)
7. [Hardware Implementation](#hardware-implementation)
8. [GUI Implementation](#gui-implementation)
9. [Conclusion](#conclusion)

## Goals and Workflow
### Goals
- Develop a robust anti-spoofing detection system.
- Ensure high accuracy in distinguishing between genuine and spoofed images using a pre-trained Support Vector Machine (SVM) model.
- Implement a Python-based GUI for better visual understanding.
- Achieve software-to-hardware implementation on a Raspberry Pi 5.

### Workflow
1. **Data Collection**
   - Collection of images from 233 individuals, each contributing 12 images displaying various angles, including caps and spectacles.
   - Generation of counterfeit images by manually capturing screenshots of the original images.
   
2. **Data Preprocessing**
   - Utilization of Multi-task Cascaded Convolutional Neural Networks (MTCNN) for face and facial landmark detection.

3. **Feature Extraction**
   - Extraction of features using various models like FaceNet, LBP, HOG, VGG16, VGG19.

4. **Training and Testing**
   - Creation of training and testing folders.
   - Selection of images for training and testing.
   - Training and testing of SVM with a mix of original and fake images.
   - Adoption of transfer learning approach using DenseNet model for enhanced performance.

5. **Performance Evaluation**
   - Evaluation of the model’s performance, particularly focusing on its accuracy in identifying fake images.
   - Transition to real-time live data testing to ensure consistency and robustness.

6. **Hardware Implementation**
   - Implementation on Raspberry Pi, including hardware and software setup, training, testing, and addressing implementation challenges.

7. **GUI Implementation**
   - Development of a graphical user interface using Python and Tkinter.
   - Integration of the GUI with the system for practical use.

## Dataset Description
### Antispoofing Dataset
- **Total Folders**: 233
- **Each Folder Contains**:
  - `original` (12 images)
  - `fake` (12 images)

### Facial Recognition Dataset
- **Total Folders**: 233
- **Each Folder Contains**:
  - 12 images per folder

## Data Preprocessing
1. **Resizing**: Images resized to 224x224 pixels.
2. **Normalization**: Pixel values normalized to the [0, 1] range.

## Feature Extraction Models

| Model   | Image Size (Face) | Image Size (Whole) | Feature Vector Dimension | Color Mode | Avg. Time (Face) | Avg. Time (Whole) |
|---------|-------------------|--------------------|--------------------------|------------|------------------|-------------------|
| FaceNet | (100, 100, 3)     | (100, 100, 3)      | 512                      | RGB        | 0.081 s          | 0.101 s           |
| LBP     | (100, 100)        | (100, 100)         | 256                      | Gray-Scale | 0.109 s          | 0.163 s           |
| HOG     | (100, 100)        | (100, 100)         | 288                      | Gray-Scale | 0.0339 s         | 0.407 s           |
| VGG16   | (100, 100, 3)     | (100, 100, 3)      | 512                      | RGB        | 0.406 s          | 0.366 s           |
| VGG19   | (100, 100, 3)     | (100, 100, 3)      | 512                      | RGB        | 0.477 s          | 0.228 s           |

## Training and Testing Protocol
1. **Folder Creation**: New folders created for training and testing.
2. **Image Selection**: 
   - 6 images from each `original` and `fake` subfolder for training (total 12 images per folder).
   - 6 images from each `original` and `fake` subfolder for testing (total 12 images per folder).
3. **SVM Training**: Uses 6 `original` and 6 `fake` images.
4. **SVM Testing**: Uses remaining 6 `original` and 6 `fake` images.
5. **Adoption of Transfer Learning**: Using DenseNet model for improved real-time performance.

## Performance and Evaluation
### Antispoofing - Uncropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 152             | 83.388  | 90.625 | 97.69 | 97.64  | 85.47  |
| 233             | 74.356  | 91.309 | 97.818| 95.708 | 96.352 |

### Antispoofing - Cropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 152             | 63.158  | 91.064 | 86.45 | 76.75  | 70.011 |
| 233             | 60.30   | 91.881 | 87.232| 92.239 | 65.379 |

### Facial Recognition - Uncropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 30              | 70.00   | 63.889 | 76.66 | 93.889 | 94.444 |
| 60              | 71.667  | 62.778 | 76.38 | 95.556 | 94.722 |
| 152             | 51.974  | 69.298 | 80.92 | 96.930 | 97.259 |
| 233             | 41.732  | 64.306 | 78.239| 95.422 | 93.772 |

### Facial Recognition - Cropped Data

| No. of Subjects | FaceNet | LBP    | HOG   | VGG16  | VGG19  |
|-----------------|---------|--------|-------|--------|--------|
| 30              | 100.00  | 19.444 | 29.44 | 41.111 | 26.111 |
| 60              | 100.00  | 16.389 | 29.72 | 31.389 | 20.833 |
| 152             | 98.684  | 16.228 | 21.139| 21.162 | 11.732 |
| 233             | 90.272  | 14.235 | 20.974| 20.815 | 8.948  |

### DenseNet Performance

![Screenshot 2024-06-17 225130](https://github.com/magnus-6/projects_dis/assets/121368258/2d810ad8-76bd-48b5-a953-1ed73ab2b9b8)


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

For more detailed information on each section, please refer to the respective chapters in the main document.
