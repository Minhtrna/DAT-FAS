import tensorflow as tf
import cv2
import numpy as np

def generate_FT(image):
    """
    Generate Fourier Transform image from input image.
    Args:
        image: Input image (numpy array, BGR or RGB). 
               Note: The original implementation converts BGR to GRAY.
               If input is RGB (from TF decode), we should handle it.
               The original code assumes cv2.imread which is BGR.
    Returns:
        fimg: Fourier Transform image (numpy array), float32, normalized.
    """
    # Check if image is a tensor, if so, we shouldn't be here or it should be wrapped.
    # This function is intended to be wrapped by tf.numpy_function or used eagerly.
    
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Handle RGB/BGR. 
    # If the pipeline loads RGB (common in TF), and we need same behavior as original 
    # (which likely used BGR from cv2.imread), we need to ensure conversion is correct.
    # Original: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # If we assume input might be RGB, we should use COLOR_RGB2GRAY.
    # For safety, let's assume the user will handle channel ordering or we standardise on RGB in TF.
    # Using RGB2GRAY is safer for standard TF pipelines.
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if it looks like BGR or RGB? hard to tell. Assumes RGB for TF compatibility.
        # But if we want exact reproduction of "cv2.imread -> cv2.cvtColor(..., BGR2GRAY)"
        # and we load with tf.io.decode_image (RGB), then:
        # RGB -> Gray is what we want.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        # Fallback or error
        gray = image
        
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    
    # Optimization of the original loop for min/max
    # Original implementation:
    # maxx = -1
    # minn = 100000
    # for i in range(len(fimg)): ...
    # This was finding global min/max.
    
    minn = np.min(fimg)
    maxx = np.max(fimg)
    
    if maxx - minn == 0:
        return np.zeros_like(fimg)
        
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    
    # Original returns fimg. It's float64 by default from numpy, usually we want float32 in TF.
    return fimg.astype(np.float32)

def generate_FT_tf(image_tensor):
    """
    TensorFlow wrapper for generate_FT.
    Args:
        image_tensor: TF Tensor (H, W, 3) or (H, W)
    Returns:
        TF Tensor (H, W, 1) or (H, W) depending on requirements.
        Original produced (H, W) I believe, but dataset_folder.py did unsqueeze(0).
    """
    ft_img = tf.numpy_function(generate_FT, [image_tensor], tf.float32)
    # Shape inference is lost with numpy_function, set it explicitely to Rank 2 (H, W)
    ft_img.set_shape([None, None])
    return ft_img

def read_image(path):
    """
    Read image using TensorFlow OPS.
    Returns RGB image [0, 255] uint8.
    """
    image_string = tf.io.read_file(path)
    # Don't use decode_image as it doesn't return shape info sometimes, use decode_jpeg or decode_png
    # or generic wrapper that sets shape.
    image = tf.io.decode_image(image_string, channels=3, expand_animations=False) 
    image.set_shape([None, None, 3])
    return image
