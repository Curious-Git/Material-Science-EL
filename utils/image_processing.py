import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import io

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input
    
    Args:
        image: PIL Image object
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        np.array: Processed image array ready for model input
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        # Resize the image
        image = image.resize(target_size)
        img_array = np.array(image)
    else:
        # If it's already a numpy array
        img_array = image
        img_array = cv2.resize(img_array, target_size)
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Normalize the image
    img_array = img_array.astype(np.float32) / 255.0
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def apply_filters(image, filters):
    """
    Apply various image processing filters
    
    Args:
        image: PIL Image object
        filters: Dict of filter names and boolean values to apply
        
    Returns:
        PIL.Image: Processed image
    """
    # Make a copy to avoid modifying the original
    processed_img = image.copy()
    
    # Convert to grayscale if selected
    if filters.get("Grayscale Conversion", False):
        processed_img = processed_img.convert("L").convert("RGB")
    
    # Enhance contrast if selected
    if filters.get("Contrast Enhancement", False):
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(1.5)  # Enhance contrast by factor of 1.5
    
    # Apply noise reduction if selected
    if filters.get("Noise Reduction", False):
        processed_img = processed_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Apply edge detection if selected
    if filters.get("Edge Detection", False):
        # Convert to numpy for OpenCV processing
        img_np = np.array(processed_img)
        
        # Convert to grayscale for edge detection
        if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert back to PIL
        processed_img = Image.fromarray(edges).convert("RGB")
    
    return processed_img

def overlay_defects(image, defects):
    """
    Overlay detected defects on the image
    
    Args:
        image: PIL Image object
        defects: List of dictionaries containing defect information
        
    Returns:
        PIL.Image: Image with defects overlaid
    """
    # Convert PIL image to numpy array for OpenCV processing
    img_np = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Draw defects on the image
    for defect in defects:
        # Get defect location
        x, y, w, h = defect["location"]
        
        # Draw rectangle
        color = (255, 0, 0)  # Red for defects
        thickness = 2
        cv2.rectangle(img_np, (x, y), (x + w, y + h), color, thickness)
        
        # Add defect type and confidence
        defect_text = f"{defect['defect_type']}: {defect['confidence']:.2f}"
        cv2.putText(img_np, defect_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # Convert back to PIL image
    return Image.fromarray(img_np)

def create_comparison_plot(original, processed, figsize=(12, 6)):
    """
    Create a comparison plot of original and processed images
    
    Args:
        original: Original PIL Image
        processed: Processed PIL Image
        figsize: Size of the figure
        
    Returns:
        bytes: Plot as a byte stream
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(np.array(original))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot processed image
    axes[1].imshow(np.array(processed))
    axes[1].set_title("Processed Image")
    axes[1].axis("off")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to byte stream
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    plt.close(fig)
    
    return buf
