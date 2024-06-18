import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

# Initialize VGG16 model for feature extraction
vgg16 = VGG16(weights='imagenet', include_top=False)

# Set up the PiCamera with specific resolution and framerate
camera = PiCamera()
camera.resolution = (512, 304)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(512, 304))

# Function to capture an image
def capture_image():
    rawCapture.truncate(0)
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    rawCapture.truncate(0)
    return image

# Capture an image
captured_image = capture_image()

# Preprocess the image for VGG16
resized_image = cv2.resize(captured_image, (224, 224))
preprocessed_image = preprocess_input(resized_image)
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

# Extract feature vectors using VGG16
features = vgg16.predict(preprocessed_image)

print("Feature vectors extracted:", features)

# Cleanup and release resources
camera.close()
cv2.destroyAllWindows()
