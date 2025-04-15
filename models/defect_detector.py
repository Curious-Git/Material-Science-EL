import tensorflow as tf
import numpy as np
from utils.model_utils import load_model, create_dummy_model
from utils.image_processing import preprocess_image, overlay_defects
import os
import random

class DefectDetector:
    """Class for detecting defects in material microstructure images"""
    
    def __init__(self, model_name="ResNet50"):
        """
        Initialize the defect detector
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        
        # Define defect classes
        self.defect_types = [
            "Crack",
            "Void",
            "Inclusion",
            "Porosity",
            "Phase Segregation",
            "Grain Boundary Issue"
        ]
        
        # Number of classes (including no defect)
        self.num_classes = len(self.defect_types) + 1
        
        # Initialize the model
        try:
            self.model = load_model(model_name, self.num_classes)
        except Exception as e:
            print(f"Failed to load real model: {e}")
            print("Using dummy model instead")
            self.model = create_dummy_model(model_name, self.num_classes)
    
    def detect(self, image):
        """
        Detect defects in the image
        
        Args:
            image: PIL Image object or preprocessed numpy array
            
        Returns:
            tuple: (defects_list, image_with_defects_highlighted)
        """
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # In a real implementation, a proper object detection model would be used
        # For demo purposes, we'll simulate defect detection with random boxes
        
        # Get image dimensions
        width, height = image.size
        
        # Create a list to store detected defects
        defects = []
        
        # For demo: create 0-3 random defects
        num_defects = random.randint(0, 3)
        
        for _ in range(num_defects):
            # Random location (x, y, width, height)
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            w = random.randint(20, 50)
            h = random.randint(20, 50)
            
            # Random defect type and confidence
            defect_type = random.choice(self.defect_types)
            confidence = random.uniform(0.6, 0.95)
            
            # Add to defects list
            defects.append({
                "defect_type": defect_type,
                "confidence": confidence,
                "location": (x, y, w, h)
            })
        
        # Create an image with the defects highlighted
        image_with_defects = overlay_defects(image, defects)
        
        return defects, image_with_defects
    
    def has_critical_defects(self, defects, critical_threshold=0.8):
        """
        Check if the image has critical defects
        
        Args:
            defects: List of defect dictionaries
            critical_threshold: Confidence threshold for critical defects
            
        Returns:
            bool: True if critical defects found, False otherwise
        """
        critical_defect_types = ["Crack", "Void"]
        
        for defect in defects:
            if (defect["defect_type"] in critical_defect_types and 
                defect["confidence"] >= critical_threshold):
                return True
        
        return False
    
    def get_defect_statistics(self, defects):
        """
        Get statistics about the detected defects
        
        Args:
            defects: List of defect dictionaries
            
        Returns:
            dict: Dictionary with defect statistics
        """
        if not defects:
            return {"count": 0, "types": {}}
        
        # Count defects by type
        defect_counts = {}
        for defect in defects:
            defect_type = defect["defect_type"]
            if defect_type in defect_counts:
                defect_counts[defect_type] += 1
            else:
                defect_counts[defect_type] = 1
        
        # Calculate average confidence
        avg_confidence = sum(d["confidence"] for d in defects) / len(defects)
        
        return {
            "count": len(defects),
            "types": defect_counts,
            "average_confidence": avg_confidence
        }
