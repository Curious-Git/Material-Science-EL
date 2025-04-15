import tensorflow as tf
import numpy as np
from utils.model_utils import load_model, create_dummy_model
from utils.image_processing import preprocess_image
import os

class MaterialClassifier:
    """Class for classifying material types from microstructure images"""
    
    def __init__(self, model_name="ResNet50"):
        """
        Initialize the material classifier
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        
        # Define material classes
        self.class_names = [
            "Carbon Steel",
            "Stainless Steel",
            "Aluminum Alloy",
            "Copper Alloy",
            "Titanium Alloy",
            "Cast Iron",
            "Nickel Alloy"
        ]
        
        # Number of classes
        self.num_classes = len(self.class_names)
        
        # Initialize the model
        try:
            self.model = load_model(model_name, self.num_classes)
        except Exception as e:
            print(f"Failed to load real model: {e}")
            print("Using dummy model instead")
            self.model = create_dummy_model(model_name, self.num_classes)
        
    def predict(self, image):
        """
        Predict the material type from the image
        
        Args:
            image: PIL Image object or preprocessed numpy array
            
        Returns:
            dict: Dictionary mapping material types to confidence scores
        """
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predictions = predictions[0]  # Get the first (and only) prediction
        
        # Map probabilities to class names
        result = {self.class_names[i]: float(predictions[i]) for i in range(len(self.class_names))}
        
        # Sort the results by confidence (highest first)
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        
        return result
    
    def get_top_prediction(self, image, threshold=0.5):
        """
        Get the top material prediction if it's above the confidence threshold
        
        Args:
            image: PIL Image object or preprocessed numpy array
            threshold: Confidence threshold
            
        Returns:
            tuple: (material_type, confidence) or (None, 0) if no prediction above threshold
        """
        # Get all predictions
        predictions = self.predict(image)
        
        # Get the top prediction
        top_material = next(iter(predictions))
        top_confidence = predictions[top_material]
        
        # Check if above threshold
        if top_confidence >= threshold:
            return (top_material, top_confidence)
        else:
            return (None, 0)
