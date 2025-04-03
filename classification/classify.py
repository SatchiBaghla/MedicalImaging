from PIL import Image
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import os

def predict(img, st):
    model_path = os.path.join('.', 'runs', 'classify', 'train', 'weights', 'best.pt')
    
    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Preprocess the input image
    try:
        # If the input is a PIL.Image object
        if isinstance(img, Image.Image):
            img = np.array(img)  # Convert to a NumPy array
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Convert RGB to BGR

        # If the input is already a NumPy array
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:  # Grayscale image
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA image
                img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)

        else:
            raise ValueError("Input image must be a PIL.Image or NumPy array")

        # Check if the image has the correct shape (3 channels)
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("Input image must be in BGR format with three channels")

        print(f"Image shape before prediction: {img.shape}")

        
        # Run YOLOv8 prediction
        results = model.predict(img)

        # Convert generator results to a list
        results_list = list(results)

        # Ensure predictions exist
        if len(results_list) == 0:
            raise ValueError("No predictions were made. Check the input image.")

        # Get the first result
        result = results_list[0]

        # Extract class names and probabilities
        class_names = result.names
        probs = result.probs.data.tolist()
        class_name = class_names[np.argmax(probs)].upper()

        # Draw the predicted class name on the image
        height, width = img.shape[:2]
        cv.putText(img, class_name, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # Convert the image back to RGB for Streamlit display
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.subheader('Output Image')
        st.image(img_rgb, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


# Load and test the function with an image
#image_path = "classification_screenshot.png"  # Change to your image path
#img = cv.imread(image_path)

#if img is None:
   # print(f"Error: Unable to load image from {image_path}")
#else:
   # predict(img)