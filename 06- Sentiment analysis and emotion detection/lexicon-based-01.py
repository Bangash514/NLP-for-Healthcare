

#lexicon-based sentiment analysis approach using the VADER and AFINN lexicons. 
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Directory containing images
image_dir = r"E:/PhD 2022-26/陕西师范大学/Second Year 2024-25/ExperimentsforImages/HAM10000_images_part_1"

# Function to preprocess images
def preprocess_images(image_directory):
    images = []
    for img_name in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_name)
        
        # Load and resize image to 128x128
        try:
            img = load_img(img_path, target_size=(128, 128))  # Load image
            img_array = img_to_array(img)  # Convert image to numpy array
            images.append(img_array)  # Append image to list
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
    
    return np.array(images)  # Return as numpy array

# Run the preprocessing function and load images
try:
    image_data = preprocess_images(image_dir)
    print(f"Loaded {len(image_data)} images with shape {image_data.shape}")
except FileNotFoundError:
    print("The specified directory does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")
