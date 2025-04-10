import os
import glob
import gc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
import random

image_dir = "/kaggle/input/aaltoescv/train/train/images"
mask_dir = "/kaggle/input/aaltoescv/train/train/masks"
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4

image_paths = sorted([os.path.join(image_dir, f)
                     for f in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true > 0.5, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + \
        K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return (intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred)

def load_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE])
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask


def make_dataset(img_paths, mask_paths):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_imgs, train_masks)
val_ds = make_dataset(val_imgs, val_masks)

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs)

    skips = [
        base_model.get_layer("block2a_expand_activation").output,
        base_model.get_layer("block3a_expand_activation").output,
        base_model.get_layer("block4a_expand_activation").output,
        base_model.get_layer("block6a_expand_activation").output,
    ]
    x = base_model.output

    for i, skip in reversed(list(enumerate(skips))):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, 128 if i > 1 else 64)

    x = layers.UpSampling2D()(x)
    x = conv_block(x, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)

def load_latest_final_model():
    models_list = glob.glob("final_model_*_iou_*.keras")
    if not models_list:
        print("No final model found — using current model as-is.")
        return None
    latest_model = max(models_list, key=os.path.getctime)
    print(f"Loading latest model: {latest_model}")
    return load_model(latest_model, custom_objects={"iou_score": iou_score})


model = build_unet()
model = load_latest_final_model() or model

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=combined_loss,
    metrics=["accuracy", iou_score]
)

if os.path.exists("checkpoint.keras"):
    print("Resuming from checkpoint...")
    model = load_model("checkpoint.keras", custom_objects={
                       "iou_score": iou_score})
else:
    print("Starting fresh model...")

callbacks = [
    ModelCheckpoint("checkpoint.keras", monitor="val_iou_score",
                    mode="max", save_best_only=True, verbose=1),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

K.clear_session()
gc.collect()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

val_loss, val_accuracy, val_iou = model.evaluate(val_ds)
print("\n Final Validation Results:")
print(f"   Loss      : {val_loss:.4f}")
print(f"   Accuracy  : {val_accuracy:.4f}")
print(f"   IoU Score : {val_iou:.4f}")

def get_next_model_filename(iou, base_path="final_model"):
    num = 1
    while os.path.exists(f"{base_path}_{num}_iou_{iou:.4f}.keras"):
        num += 1
    return f"{base_path}_{num}_iou_{iou:.4f}.keras"

final_model_path = get_next_model_filename(val_iou)
model.save(final_model_path)
print(f"Final model saved as: {final_model_path}")

sample_indices = random.sample(range(len(val_imgs)), 5)
fig, axes = plt.subplots(len(sample_indices), 3,
                         figsize=(12, 4 * len(sample_indices)))

for i, idx in enumerate(sample_indices):
    image_path = val_imgs[idx]
    mask_path = val_masks[idx]

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (256, 256))
    mask_binary = (mask_resized > 127).astype(np.uint8)

    input_tensor = np.expand_dims(img_resized, axis=0)
    pred_mask = model.predict(input_tensor, verbose=0)[0]
    pred_binary = (pred_mask > 0.5).astype(np.uint8).squeeze()

    axes[i, 0].imshow(img_resized)
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask_binary, cmap='gray')
    axes[i, 1].set_title("Ground Truth Mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_binary, cmap='gray')
    axes[i, 2].set_title("Predicted Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
