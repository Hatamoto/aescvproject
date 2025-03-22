import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras import backend as K

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()

print(f"✅ Using GPU strategy with {strategy.num_replicas_in_sync} device(s)")

image_dir = "./dataset/train/train/images/"
mask_dir = "./dataset/train/train/masks/"

IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 32

initial_learning_rate = 1e-3  # Common starting point for Adam, can be adjusted

# Collect paths for images and masks
image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

def postprocess_mask(pred_mask):
    return (pred_mask > 0.5).astype(np.uint8) * 255  # Duotone output (0 or 255)

from tensorflow.keras import backend as K

def iou_score(y_true, y_pred, smooth=1e-6):
    # Ensure duotone (binary) format for accurate IoU calculation
    y_true = K.cast(y_true > 0.5, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou