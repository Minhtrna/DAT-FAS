import tensorflow as tf
import os
import glob
from source.data_io import tf_functional as F
from source.data_io import tf_transform as trans

def get_class_names(root_dir):
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def get_train_dataset(conf):
    """
    Create a tf.data.Dataset for training.
    Match: get_train_loader(conf)
    Returns: mapped dataset (image, ft_sample, target)
    """
    root_path = os.path.join(conf.train_root_path, conf.patch_info)
    
    # Get classes
    classes, class_to_idx = get_class_names(root_path)
    
    # We can use tf.lookup.StaticHashTable for graph-mode label lookup if we wanted pure graph,
    # but since we are doing some py logic, mapped function with py_function is easier or 
    # string manipulation in TF. 
    # Let's try string manipulation in TF for labels (Parse path).
    
    # Create lookup table
    keys_tensor = tf.constant(list(class_to_idx.keys()))
    vals_tensor = tf.constant(list(class_to_idx.values()))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1
    )

    # Transform definition
    # Note: Conf.input_size is expected to be a tuple (h, w) or int.
    # DatasetLoader uses: trans.RandomResizedCrop(size=tuple(conf.input_size), scale=(0.9, 1.1))
    
    input_size = conf.input_size if isinstance(conf.input_size, tuple) else (conf.input_size, conf.input_size)
    
    train_transform = trans.Compose([
        trans.ToPILImage(), # cast/setup
        trans.RandomResizedCrop(size=input_size, scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    
    # File pattern
    # root/class/image.ext
    # glob pattern: root/*/*
    file_pattern = os.path.join(root_path, '*', '*')
    
    # Create dataset from chunks of files to avoid slow listing if millions of files? 
    # list_files allows shuffle.
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    
    def process_path(file_path):
        # file_path is a tensor string
        
        # 1. Get Label
        # .../class/image.jpg
        parts = tf.strings.split(file_path, os.sep)
        # parts[-2] is class name.
        # But if os.sep varies or split behavior... usually works.
        label_str = parts[-2]
        label = table.lookup(label_str)
        
        # 2. Read Image
        image_bytes = tf.io.read_file(file_path)
        # Decode. 
        # Note: Original opencv_loader uses cv2.imread which ignores orientation EXIF usually, 
        # tf.io.decode_image usually respects it or not? expand_animations=False.
        # Ensure 3 channels.
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        
        # 3. Generate FT
        # Need pure image before transform for FT generation per original logic?
        # DatasetFolderFT line 29: sample = self.loader(path) -> img
        # line 31: ft_sample = generate_FT(sample)
        # line 44: sample = self.transform(sample)
        
        # So FT is generated from original loaded image.
        ft_sample = F.generate_FT_tf(image) # Returns float32 tensor
        
        # ft_sample resize
        # Original: cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        # ft_sample is (H, W). Add channel dim -> (H, W, 1)
        ft_sample = tf.expand_dims(ft_sample, -1)
        ft_sample = tf.image.resize(ft_sample, (conf.ft_height, conf.ft_width))
        
        # ft_sample processing
        # Original: torch.from_numpy(ft_sample).float().unsqueeze(0) -> (1, H, W)
        # TF: Keep (H, W, 1) usually. 
        # If we really need (1, H, W)...
        # Let's keep (H, W, 1) (Channels last).
        
        # 4. Transform Image
        # ToPILImage does essentially nothing in our TF impl (just cast), 
        # but the inputs to transforms expect (H, W, 3).
        image = train_transform(image)
        
        return image, ft_sample, label

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(conf.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
