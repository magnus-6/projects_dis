import joblib
import RPi.GPIO as GPIO
import time
import numpy as np
import sys  # Import sys to use sys.exit()

# Load the pre-trained SVM model for anti-spoofing
svm_antispoofing_model = joblib.load("svm_models/svm_antispoofing.joblib")

# Setup LED GPIO for signaling
RED_PIN = 18
GREEN_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

# Function to turn on and off the LEDs
def signal_red(duration=2):
    GPIO.output(RED_PIN, GPIO.HIGH)
    time.sleep(duration)  # Keep the red light on for a specified duration
    GPIO.output(RED_PIN, GPIO.LOW)

def signal_green():
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(GREEN_PIN, GPIO.LOW)



# Predict with the SVM anti-spoofing model
classification = svm_antispoofing_model.predict([features])

# If classified as '0', light up red LED for 10 seconds and exit
if classification[0] == 0:
    signal_red(10)  # Red light for 10 seconds
    print("Classification result: 0 (Spoofing/Not permitted)")
    sys.exit()  # Exit the entire script
# If classified as '1', light up green LED (indicating real or permitted)
else:
    signal_green()  # Green light for 2 seconds
    print("Classification result: 1 (Real/Permitted)")
