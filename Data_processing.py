"""
X-RAY PREPROCESSING PIPELINE (TensorFlow + OpenCV)
!! REPLACE PATHS MARKED WITH # <-- !!
"""
import cv2
import tensorflow as tf
import numpy as np
import os

# ======================
# 1. PATH CONFIGURATION
# ======================
INPUT_DIR = "/home/madhavr/Yantragya/Yantragya_pramukh_vatavaran/Bone_Age_data/BoneAge_Dataset/Training_dataset_BoneAge"  # <-- REPLACE WITH YOUR RAW IMAGE PATH
OUTPUT_DIR = "/home/madhavr/Yantragya/Yantragya_pramukh_vatavaran/Processed_datasets/processed_xray.tfrecord"   # <-- SET PROCESSED OUTPUT PATH
IMG_SIZE = (224, 224)  # Standard size for medical imaging models

# ======================
# 2. CORE PROCESSING
# ======================
def enhance_contrast(img):
    """CLAHE contrast enhancement (OpenCV)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

def preprocess_xray(img_path):
    """Main processing pipeline"""
    # 1. Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Basic enhancements
    img = enhance_contrast(img)
    
    # 3. Adaptive thresholding [9]
    _, mask = cv2.threshold(img, 0, 255, 
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Apply mask and resize
    processed = cv2.bitwise_and(img, img, mask=mask)
    processed = cv2.resize(processed, IMG_SIZE)
    
    # 5. Normalize for TensorFlow
    return processed.astype(np.float32) / 255.0

# ======================
# 3. TENSORFLOW PIPELINE
# ======================
def create_dataset(batch_size=32):
    """TensorFlow data pipeline"""
    # Get image paths
    img_paths = [os.path.join(INPUT_DIR, f) 
                for f in os.listdir(INPUT_DIR) 
                if f.lower().endswith(('.png', '.jpg'))]
    
    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    
    # Map preprocessing
    def tf_preprocess(path):
        def process_fn(p):
            return preprocess_xray(p.numpy().decode())
        return tf.py_function(process_fn, [path], tf.float32)
    
    return dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                 .batch(batch_size) \
                 .prefetch(buffer_size=tf.data.AUTOTUNE)

# ======================
# 4. SAVE PROCESSED DATA
# ======================
def save_processed(dataset, output_path):
    """Save as TFRecord for model training"""
    def _bytes_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_path) as writer:
        for batch in dataset:
            for img in batch:
                # Serialize image
                img_bytes = tf.io.serialize_tensor(img).numpy()
                
                # Create example
                feature = {'image': _bytes_feature(img_bytes)}
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                
                writer.write(example.SerializeToString())

# ======================
# 5. EXECUTION
# ======================
if __name__ == "__main__":
    # 1. Create dataset
    ds = create_dataset()
    
    # 2. Save processed data
    save_processed(ds, OUTPUT_DIR)
    print(f"Processed X-rays saved to {OUTPUT_DIR}")
