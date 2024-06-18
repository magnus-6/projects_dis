import pandas as pd
import numpy as np
import joblib  # Import joblib for model saving
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Function to convert the 'feature_Vector' string to a list of floats
def parse_feature_vector(feature_str):
    # Remove unwanted characters and split into a list of floats
    return list(map(float, feature_str.strip('[]').split(',')))

# Read the CSV files
train_embeddings_df = pd.read_csv('train_embeddings.csv')
test_embeddings_df = pd.read_csv('test_embeddings.csv')

# Extract the first column ('feature_Vector') and convert it to numpy arrays
trainX = np.array([parse_feature_vector(row) for row in train_embeddings_df.iloc[:, 0]])
testX = np.array([parse_feature_vector(row) for row in test_embeddings_df.iloc[:, 0]])

# Extract the third column ('Classification') as the target variable
trainy = train_embeddings_df.iloc[:, 2].values
testy = test_embeddings_df.iloc[:, 2].values

# Standardize the feature vectors
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Create SVM model with hyperparameter tuning using GridSearchCV
svm_pipeline = Pipeline([
    ("normalizer", Normalizer(norm='l2')),  # Normalization step
    ("svm", SVC(probability=True))  # SVM classifier with probability
])

# Hyperparameter grid for tuning
param_grid = {
    "svm__kernel": ["rbf"],
    "svm__C": [0.1, 1, 10],  # Regularization strength
    "svm__gamma": ["scale", "auto"],  # Gamma value
}

# Create GridSearchCV object with cross-validation
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(trainX, trainy)

# Make predictions with the best model
best_model = grid_search.best_estimator_
yhat_train_svm = best_model.predict(trainX)
yhat_test_svm = best_model.predict(testX)

# Calculate accuracy
accuracy_train_svm = accuracy_score(trainy, yhat_train_svm)
accuracy_test_svm = accuracy_score(testy, yhat_test_svm)

# Display best hyperparameters and accuracy
print("Best Hyperparameters:", grid_search.best_params_)
print('SVM Accuracy: train=%.3f, test=%.3f' % (accuracy_train_svm * 100, accuracy_test_svm * 100))

# Save the trained SVM model to a file using joblib
joblib.dump(best_model, 'svm_model.joblib')
