import tensorflow as tf
import numpy as np
import os
from PIL import Image

def load_model(model_name, num_classes, input_shape=(224, 224, 3)):
    """
    Load a pre-trained model and adapt it for our use case
    
    Args:
        model_name: Name of the pre-trained model
        num_classes: Number of classes for classification
        input_shape: Input shape for the model
        
    Returns:
        tf.keras.Model: The loaded and adapted model
    """
    # Dictionary mapping model names to their TF implementation
    model_dict = {
        "ResNet50": tf.keras.applications.ResNet50,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0
    }
    
    # Get the model constructor
    model_constructor = model_dict.get(model_name)
    
    if model_constructor is None:
        raise ValueError(f"Model {model_name} not supported")
    
    # Load the pre-trained model without the classification layer
    base_model = model_constructor(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification layers
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the new model
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)
    
    return model

def predict_with_model(model, image, class_names):
    """
    Make predictions using the provided model
    
    Args:
        model: tf.keras.Model
        image: Preprocessed image array
        class_names: List of class names
        
    Returns:
        dict: Dictionary mapping class names to their confidence scores
    """
    # Make prediction
    predictions = model.predict(image)
    predictions = predictions[0]  # Get the first (and only) prediction
    
    # Map probabilities to class names
    result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    
    return result

def create_dummy_model(model_name, num_classes):
    """
    Create a simple dummy model for testing purposes when pre-trained models are not available
    
    Args:
        model_name: Name of the model (just for consistency)
        num_classes: Number of classes for classification
        
    Returns:
        object: A dummy model object with predict method
    """
    class DummyModel:
        def __init__(self, name, num_classes):
            self.name = name
            self.num_classes = num_classes
        
        def predict(self, image):
            # Generate random predictions
            batch_size = image.shape[0]
            preds = np.random.rand(batch_size, self.num_classes)
            # Normalize to sum to 1
            preds = preds / preds.sum(axis=1, keepdims=True)
            return preds
    
    return DummyModel(model_name, num_classes)
