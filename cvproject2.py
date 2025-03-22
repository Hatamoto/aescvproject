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

mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
print(f"✅ Using GPU strategy with {strategy.num_replicas_in_sync} device(s)")

image_dir = "./dataset/train/train/images/"
mask_dir = "./dataset/train/train/masks/"

IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 32
initial_learning_rate = 1e-3

image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

def postprocess_mask(pred_mask):
    return (pred_mask > 0.5).astype(np.uint8) * 255

from tensorflow.keras import backend as K

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true > 0.5, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def improved_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return 1 - dice

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)

    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    tversky_loss_value = tversky_loss(y_true, y_pred)

    return 0.1 * dice_loss + 0.7 * tversky_loss_value + 0.2 * bce_loss

def unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.3)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    b1 = layers.Conv2D(256, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    b1 = layers.BatchNormalization()(b1)

    u1 = layers.Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='same')(b1)
    u1 = layers.concatenate([u1, c1])
    u1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.Dropout(0.3)(u1)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u1)
    model = keras.Model(inputs, outputs)

    return model

with strategy.scope():
    model = unet_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=combined_loss,
        metrics=['accuracy', iou_score]
    )
