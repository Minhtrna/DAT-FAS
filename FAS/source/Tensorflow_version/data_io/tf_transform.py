import tensorflow as tf
import numpy as np
import cv2
import math

class Compose(object):
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor(object):
    """Convert ndarray/tensor to float tensor [0, 255] (H, W, C)."""
    def __call__(self, pic):
        # Assumes pic is (H, W, C)
        return tf.cast(pic, tf.float32)

class ToPILImage(object):
    """Placeholder for ToPILImage, in TF mainly just cast to uint8 if needed."""
    def __call__(self, pic):
        return tf.cast(pic, tf.uint8)

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor: (H, W, C)
        # mean, std: lists or tuples
        return (tensor - self.mean) / self.std

class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a probability of 0.5."""
    def __call__(self, img):
        return tf.image.random_flip_left_right(img)

class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio."""
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        # Img shape (H, W, C)
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]
        area = tf.cast(width * height, tf.float32)
        
        # We need to implement the loop logic using TF or just python if mapped
        # For simplicity in graph mode, we might approximate or use py_function.
        # But here let's try to use python logic assuming eager/py_func wrapping for random generation 
        # or implement a simple version.
        # The original loop tries 10 times.
        
        # Helper to generate random crop params
        def gen_random_crop():
             # Logic matching PyTorch's RandomResizedCrop
             target_area = np.random.uniform(*scale) * float(area)
             aspect_ratio = np.random.uniform(*ratio)
             
             w = int(round(math.sqrt(target_area * aspect_ratio)))
             h = int(round(math.sqrt(target_area / aspect_ratio)))
             
             if np.random.random() < 0.5:
                 w, h = h, w
                 
             if w <= width and h <= height:
                 i = np.random.randint(0, height - h + 1)
                 j = np.random.randint(0, width - w + 1)
                 return i, j, h, w
             return None

        # Try 10 times in python (inside py_function)
        # If we are in graph mode, this won't work directly inside the graph construction unless wrapped.
        # However, for tf.data, we are fine using numpy/python logic if we rely on it.
        
        # Simplified TF version using tf.image.random_crop is different (fixed size).
        # We want RandomResizedCrop.
        # tf.image.sample_distorted_bounding_box is the TF equivalent.
        
        return None 

    def __call__(self, img):
        # shape (H, W, C)
        shape = tf.shape(img)
        # tf.image.sample_distorted_bounding_box expects [batch, h, w, c]
        img_expanded = tf.expand_dims(img, 0)
        
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(img),
            bounding_boxes=[[[0.0, 0.0, 1.0, 1.0]]],
            min_object_covered=0.1, # Approx
            aspect_ratio_range=self.ratio,
            area_range=(self.scale[0], min(self.scale[1], 1.0)),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True
        )
        
        # Crop
        cropped = tf.slice(img, begin, size)
        
        # Resize
        resized = tf.image.resize(cropped, self.size)
        return resized

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if self.brightness > 0:
            img = tf.image.random_brightness(img, max_delta=self.brightness)
        if self.contrast > 0:
            img = tf.image.random_contrast(img, lower=max(0, 1-self.contrast), upper=1+self.contrast)
        if self.saturation > 0:
            img = tf.image.random_saturation(img, lower=max(0, 1-self.saturation), upper=1+self.saturation)
        if self.hue > 0:
            img = tf.image.random_hue(img, max_delta=self.hue)
        return img

class RandomRotation(object):
    """Rotate the image by angle."""
    def __init__(self, degrees):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, img):
        # Wrapper for cv2 rotation via numpy_function
        def rotate_py(image, min_deg, max_deg):
            angle = np.random.uniform(min_deg, max_deg)
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Use appropriate fill value? Original uses defaults (0).
            rotated = cv2.warpAffine(image, M, (w, h))
            return rotated

        # Note: input img is Tensor
        img_rotated = tf.numpy_function(
            rotate_py, 
            [img, self.degrees[0], self.degrees[1]], 
            tf.float32 if img.dtype == tf.float32 else tf.uint8 # preserve dtype
        )
        img_rotated.set_shape(img.shape)
        return img_rotated
