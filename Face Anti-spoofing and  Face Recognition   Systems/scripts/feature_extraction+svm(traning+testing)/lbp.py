

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from skimage.feature import local_binary_pattern
import time
import pandas as pd
import os
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def lbp_feature_extraction(image,height,width):
    img_lbp = np.zeros((height, width),np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(image, i, j)
            
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])        
    fd=hist_lbp
    # print(np.ravel(hist_lbp))
    fd = np.ravel(fd)
    fd = fd.reshape(1, -1)
    fd=preprocessing.normalize(fd)
    # print(fd.shape)
    return fd

# Path to the "preprocessed" folder
base_dir = "E:\DataSet"
# base_dir = "E:\MACHINE LEARNING\output2"

output_dir = "E:\MACHINE LEARNING\CSV_Final_100"
# base_dir = input("enter the address of folder where datas are stored")

#code for calculation of time

start = time.time()
timetot=[]
timet = ["No. of images" , "Time Taken"]
timetot.append(timet)

counter = add = 30
j=0

# Initialize lists to store face lbp_features and corresponding labels for training and testing
train_lbp_features = []
train_labels = []
test_lbp_features = []
test_labels = []

radius=2
numpoints=16
size1=size2=100

z=1
no=152

original = "original"

for subdir in os.listdir(base_dir):
    if(os.path.exists(os.path.join(base_dir,subdir,"Original"))):
       os.rename(os.path.join(base_dir,subdir,"Original"), os.path.join(base_dir,subdir,"original"))

    if(os.path.exists(os.path.join(base_dir,subdir,"Fake"))):
       os.rename(os.path.join(base_dir,subdir,"Fake"), os.path.join(base_dir,subdir,"fake"))    
    # for i in range(30):
    if(z<=no):
        z+=1
        subdir_path = os.path.join(base_dir, subdir,original)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            image_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)]
            
            # split image paths into training and testing datasets with a 50-50 ratio
            train_paths, test_paths = train_test_split(image_paths, test_size=0.5, random_state=42)
            
            # Initialize embedding lists for training and testing
            train_lbp_features_subfolder = []
            test_lbp_features_subfolder = []
            
            # Embed images for training
            for image_path in train_paths:
                image_array = cv2.imread(image_path)
                # image_array= cv2.resize(image_array, dsize=[size1,size2])
                image_array= cv2.resize(image_array, dsize=[100,100])
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                
                # image_size = image_gray.shape
                
                # lbp = mahotas.features.lbp(image_gray,12,9)[0]
                
                # Embed the image and append to training lbp_features
                # lbp = local_binary_pattern(image_gray, numpoints, radius)
                lbp = lbp_feature_extraction(image_gray, height= len(image_gray), width=len(image_gray[0]))
                
                # width = len(lbp)
                # height = len(lbp[0])
                # size1=width
                # size2=height
                
                # size1 = lbp.shape[0]
                # size2 = lbp.shape[1]
                
                # print(lbp.shape)
                # lbp = tf.reshape(lbp,[size1*size2]).numpy()
                
                # lbp = lbp[:512]
                # print(lbp)
                # desc=LocalBinaryPatterns(50, 20)    
                # lbp = desc.describe(image_gray)
                train_lbp_features_subfolder.append(lbp[0])
                train_labels.append(subdir)
                
                #code for calculation of time
            
                print(j)
                j=j+1
                
                if(j == counter):
                    timet=[]
                    end = time.time()
                    tott = end - start
                    timet.append(counter)
                    timet.append(tott)
                    timetot.append(timet)
                    counter += add
                    print(tott)
    
            
            # Embed images for testing
            for image_path in test_paths:
                image_array = cv2.imread(image_path)
                # image_array= cv2.resize(image_array, dsize=[size1,size2])
                image_array= cv2.resize(image_array, dsize=[100,100])
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                
                
                # Embed the image and append to training lbp_features
                # lbp = local_binary_pattern(image_gray, numpoints, radius)
                lbp = lbp_feature_extraction(image_gray, height= len(image_gray), width=len(image_gray[0]))
                # print(lbp.shape)
                
                # width = len(lbp)
                # height = len(lbp[0])
                # size1=width
                # size2=height
                
                # lbp = tf.reshape(lbp,[size1*size2]).numpy()
                
                # lbp = lbp[:512]
                # desc=LocalBinaryPatterns(50, 20)    
                # lbp = desc.describe(image_gray)
                # lbp = mahotas.features.lbp(image_gray,12,9)[0]
                test_lbp_features_subfolder.append(lbp[0])
                test_labels.append(subdir)
                
                #code for calculation of time
            
                print(j)
                j=j+1
                
                if(j == counter):
                    timet=[]
                    end = time.time()
                    tott = end - start
                    timet.append(counter)
                    timet.append(tott)
                    timetot.append(timet)
                    counter += add
                    print(tott)

        
        # Append subfolder lbp_features to the main training and testing lists
        train_lbp_features.extend(train_lbp_features_subfolder)
        test_lbp_features.extend(test_lbp_features_subfolder)
        
# Convert lbp_features and labels to DataFrames for training and testing
train_lbp_features_df = pd.DataFrame(train_lbp_features)
train_labels_df = pd.DataFrame(train_labels)
test_lbp_features_df = pd.DataFrame(test_lbp_features)
test_labels_df = pd.DataFrame(test_labels)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save lbp_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_lbp_features_{z-1}.csv")
train_lbp_features_df.to_csv(output_train_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")

# Save lbp_features and labels of the testing dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_lbp_features_{z-1}.csv")
test_lbp_features_df.to_csv(output_test_csv_path, index=False)
print(f"Testing lbp_features saved to {output_test_csv_path}")

# Save lbp_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_lbp_labels_{z-1}.csv")
train_labels_df.to_csv(output_train_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")

# Save lbp_features and labels of the training dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_lbp_labels_{z-1}.csv")
test_labels_df.to_csv(output_test_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")








import pandas as pd
import os
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# File paths
output_dir = r"E:\MACHINE LEARNING\CSV_Final_100"
z = no

# Load data
trainX = pd.read_csv(os.path.join(output_dir, f"train_lbp_features_{z}.csv"))
trainy = pd.read_csv(os.path.join(output_dir, f"train_lbp_labels_{z}.csv"))
testX = pd.read_csv(os.path.join(output_dir, f"test_lbp_features_{z}.csv"))
testy = pd.read_csv(os.path.join(output_dir, f"test_lbp_labels_{z}.csv"))

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX_normalized = in_encoder.transform(trainX)
testX_normalized = in_encoder.transform(testX)

# Label encode targets
label_encoder = LabelEncoder()
trainy_encoded = label_encoder.fit_transform(trainy)
testy_encoded = label_encoder.transform(testy)

# SVM classifier with a radial basis function kernel
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(trainX_normalized, trainy_encoded)

# Predictions
yhat_train_svm = svm_model.predict(trainX_normalized)
yhat_test_svm = svm_model.predict(testX_normalized)

# Calculate accuracies
accuracy_train_svm = accuracy_score(trainy_encoded, yhat_train_svm)
accuracy_test_svm = accuracy_score(testy_encoded, yhat_test_svm)

print('SVM Accuracy: train=%.3f, test=%.3f' % (accuracy_train_svm * 100, accuracy_test_svm * 100))





end = time.time()

timet = end-start
timei = timet/(no*12)

print(f"time is {timei}")
