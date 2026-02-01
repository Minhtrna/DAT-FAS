
import tensorflow as tf
import os
import shutil
import numpy as np
import cv2
import sys

# Add source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from source.data_io import tf_dataset_loader

def setup_dummy_data(root_dir):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)
    
    # Class 0
    c0 = os.path.join(root_dir, 'class_0')
    os.makedirs(c0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 50), (255, 255, 255), -1)
    cv2.imwrite(os.path.join(c0, 'img0.jpg'), img)
    
    # Class 1
    c1 = os.path.join(root_dir, 'class_1')
    os.makedirs(c1)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(c1, 'img1.jpg'), img)
    
    return root_dir

class Config:
    def __init__(self, root_path):
        self.train_root_path = os.path.dirname(root_path)
        self.patch_info = os.path.basename(root_path)
        self.input_size = (112, 112)
        self.ft_width = 10
        self.ft_height = 10
        self.batch_size = 2

def verify():
    test_root = 'test_dataset_root'
    abs_test_root = os.path.abspath(test_root)
    setup_dummy_data(abs_test_root)
    
    conf = Config(abs_test_root)
    print(f"Testing with root: {conf.train_root_path}, patch: {conf.patch_info}")
    
    try:
        ds = tf_dataset_loader.get_train_dataset(conf)
        print("Dataset created successfully.")
        
        for images, ft_maps, labels in ds.take(1):
            print("Batch shapes:")
            print(f"Images: {images.shape}") # Expect (2, 112, 112, 3)
            print(f"FT Maps: {ft_maps.shape}") # Expect (2, 10, 10, 1)
            print(f"Labels: {labels.shape}") # Expect (2,)
            print("Labels:", labels.numpy())
            
            # Basic sanity check
            assert images.shape == (2, 112, 112, 3)
            assert ft_maps.shape == (2, 10, 10, 1)
            
        print("Verification PASSED.")
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(abs_test_root):
             shutil.rmtree(abs_test_root)

if __name__ == "__main__":
    verify()
