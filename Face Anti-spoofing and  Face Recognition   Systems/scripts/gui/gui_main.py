
from tkinter import *
import cv2
from PIL import Image, ImageTk
from keras_facenet import FaceNet
import cv2
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import cv2
import time
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
import os

# Initialize the FaceNet embedder
mtcnn_detector = MTCNN(steps_threshold=[0.4, 0.4, 0.4])
embedder = FaceNet()

# Load the trained model
model_path = r'E:/PROJECT UPLOADS GITHUB/python gui/model_final.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("Model file not found.")

# Create a video capture object
cap = cv2.VideoCapture(0)

def detect_faces_and_extract_embeddings(captured_image):
    # input_image = cv2.imread(image_path)
    input_image = captured_image
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    faces = mtcnn_detector.detect_faces(input_image_rgb)
    
    face_embeddings = []
    for face_info in faces:
        box = face_info['box']
        x, y, w, h = box
        extracted_face = input_image[y:y+h, x:x+w]
        embedding = embedder.embeddings([extracted_face])[0]
        face_embeddings.append(embedding)
    
    return face_embeddings

# Define function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = preprocess_input(img)
    return img

# Function to perform prediction
def predict_image(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

def authentic(captured_image):
    prediction = predict_image(captured_image)
    print("Predicted Probability:", prediction)
    if prediction > 0.6:
        return 1
        # print("The image is likely genuine.")
    else:
        return 0
        # print("The image is likely spoof.")

def predict_app():
    # Your prediction logic here
    print("Predict button clicked!")
    _, frame = cap.read()
    if not _:
      print("Error capturing image")
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    
    resized_image = captured_image.resize((350, 240), Image.ANTIALIAS)  # Resize the image
    
    photo_image = ImageTk.PhotoImage(image=resized_image)
    image_widget.photo_image = photo_image
    image_widget.configure(image=photo_image)
    
    # Capture an image
    captured_image = frame    
    
    if captured_image is None:
      exit()
      
    rftype = authentic(captured_image)
    print("genuine" if rftype == 1 else "imposter")
      
    # Extract embedding from the image
    # embedding = embedder.embeddings([captured_image])[0]
    
    if(rftype == 1):
        embedding = detect_faces_and_extract_embeddings(captured_image)
    
        print("Feature vectors extracted:", embedding)
        
        # Define file paths (ensure they are correct)
        features_path = "E:/MACHINE LEARNING/test/facenet_train_features.csv"
        labels_path = "E:/MACHINE LEARNING/test/facenet_train_label.csv"
    
        # Load training embeddings and labels from CSV
    
        trainX = pd.read_csv(features_path)
        trainy = pd.read_csv(labels_path)
    
        # Prepare testing data (single sample as a list)
        testX = embedding
    
        # Normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)
    
        # Label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
    
        # SVM classifier with a radial basis function kernel
        svm_model = SVC(kernel='rbf', probability=True)
        svm_model.fit(trainX, trainy)
    
        # Make prediction on the test image
        yhat_test_svm = svm_model.predict(testX)
    
        # Decode the predicted label (assuming labels correspond to class names)
        predicted_label = out_encoder.inverse_transform(yhat_test_svm)[0]
        print(f"Predicted label: {predicted_label}")
    
        if(predicted_label == 1):
            predicted_person = "SYED ZAHEER HOSSAIN"
        elif(predicted_label == 2):
            predicted_person = "SURJIT MANDAL"
        elif(predicted_label == 3):
            predicted_person = "ANIRBAN GUHA"
        else:
            predicted_person = "PERSON IS NOT PRESENT IN OUR DATABASE"
            
        # Clear any existing labels in the text_frame
        for widget in text_frame.winfo_children():
            widget.destroy()
           
        # Create a label with some text
        display_text = "THE PERSON IN FRONT OF THE CAMERA IS " + '\n\n' + '"' + predicted_person + '"' +"\n\n" +"Image Type : Genuine"
        display_label = Label(text_frame, text=display_text, font=("Arial", 14), padx=10, pady=10)  # Customize font if needed
        display_label.config(text=display_text)
    
        # Pack the label on the window
        display_label.pack()
    else:
        # Clear any existing labels in the text_frame
        for widget in text_frame.winfo_children():
            widget.destroy()
           
        # Create a label with some text
        display_text = "Sorry! Imposters are not given Access.\n\n Show your real face to gain access."
        display_label = Label(text_frame, text=display_text, font=("Arial", 14), padx=10, pady=10)  # Customize font if needed
        display_label.config(text=display_text)
        
        # Pack the label on the window
        display_label.pack()

# Set the desired width and height
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Create a Tkinter app
app = Tk()
app.title("Person Identification Interface")
app.bind('<Escape>', lambda e: app.quit())

# Create a frame for video display
video_frame = Frame(app)
video_frame.pack(side=LEFT)

# Create a label for displaying the video
label_widget = Label(video_frame)
label_widget.pack()

# Create a blank frame on the right
blank_frame = Frame(app, width=200, height=480)
blank_frame.pack(side=RIGHT)

# Create a frame within the image frame (top part)
image_frame = Frame(blank_frame, width=400, height=240)
image_frame.pack(side=TOP)

image_widget = Label(image_frame)
image_widget.pack()

# Create another frame within the blank frame (right part)
right_frame = Frame(blank_frame, width=400, height=240)
right_frame.pack()

text_frame = Frame(right_frame, width=400, height=200)
text_frame.pack()

button_frame = Frame(right_frame, width=400, height=40 )
button_frame.pack()

predict_button = Button(button_frame, text="Predict", command=predict_app)
predict_button.pack(ipadx=5, ipady=5, expand=True)

# Function to open the camera and display frames
def open_camera():
    _, frame = cap.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)

# Call the function to start displaying frames
open_camera()

# Run the Tkinter main loop
app.mainloop()

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
