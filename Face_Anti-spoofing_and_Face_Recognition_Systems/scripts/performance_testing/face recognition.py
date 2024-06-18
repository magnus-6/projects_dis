import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

# Load training embeddings and labels from CSV
train_embeddings_df = pd.read_csv('train_embeddings_and_labels.csv')
trainX = train_embeddings_df.drop(columns=['label']).values
trainy = train_embeddings_df['label'].values

# Load testing embeddings and labels from CSV
test_embeddings_df = pd.read_csv('test_embeddings_and_labels.csv')
testX = test_embeddings_df.drop(columns=['label']).values
testy = test_embeddings_df['label'].values

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# SVM classifier with a radial basis function kernel
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(trainX, trainy)
yhat_train_svm = svm_model.predict(trainX)
yhat_test_svm = svm_model.predict(testX)
accuracy_train_svm = accuracy_score(trainy, yhat_train_svm)
accuracy_test_svm = accuracy_score(testy, yhat_test_svm)
print('SVM Accuracy: train=%.3f, test=%.3f' % (accuracy_train_svm * 100, accuracy_test_svm * 100))
