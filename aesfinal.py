import os
import cv2
import gc
import csv
import numpy as np
import tensorflow as tf
from tf import keras
from tf.keras import layers, mixed_precision, models, optimizers
from tf.keras import backend as K
from tf.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)


mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()

print(f"Using GPU strategy with {strategy.num_replicas_in_sync} device(s)")

image_dir = "./dataset/train/train/images/"
mask_dir = "./dataset/train/train/masks/"

IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16

initial_learning_rate = 1e-3

image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])

def postprocess_mask(pred_mask):
    return (pred_mask > 0.5).astype(np.uint8) * 255

def iou_score(y_true, y_pred, smooth=1e-6):

    y_true = K.cast(y_true > 0.5, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    true_pos = tf.keras.backend.sum(y_true * y_pred)
    false_neg = tf.keras.backend.sum(y_true * (1 - y_pred))
    false_pos = tf.keras.backend.sum((1 - y_true) * y_pred)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def combined_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss = 1 - (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)

    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    return 0.5 * dice_loss + 0.5 * bce_loss


with strategy.scope():
    def unet_model(input_shape=(256, 256, 3)):
        inputs = layers.Input(shape=input_shape)

        c1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Dropout(0.3)(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(128, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Dropout(0.3)(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        b1 = layers.Conv2D(256, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        b1 = layers.BatchNormalization()(b1)

        u2 = layers.Conv2DTranspose(128, (7, 7), strides=(2, 2), padding='same')(b1)
        u2 = layers.concatenate([u2, c2])
        u2 = layers.Conv2D(128, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u2 = layers.BatchNormalization()(u2)
        u2 = layers.Dropout(0.3)(u2)

        u1 = layers.Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='same')(u2)
        u1 = layers.concatenate([u1, c1])
        u1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        u1 = layers.BatchNormalization()(u1)
        u1 = layers.Dropout(0.3)(u1)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u1)

        model = models.Model(inputs, outputs)

        return model

def mask2rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle: str, label=1, shape=(256, 256)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape) 

def parse_image(image_path, mask_rle):

    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) / 255.0

    mask = tf.numpy_function(rle2mask, [mask_rle], tf.uint8)
    mask = tf.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, 1))
    mask = tf.cast(mask, tf.float32)

    return img, mask

def create_dataset(image_paths, mask_rles):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_rles))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

saved_data_path = './savedata/rle_masks.npz'

if os.path.exists(saved_data_path):
    print("Loading pre-saved data...")
    data = np.load(saved_data_path, allow_pickle=True)
    mask_rles = data['mask_rles'].tolist()
    image_paths = data['image_paths'].tolist()
    print(f"Loaded {len(image_paths)} images and {len(mask_rles)} RLE masks.")
else:
    print("No pre-saved data found — generating RLE masks...")
    mask_rles = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.uint8)
        rle_mask = mask2rle(binary_mask)
        mask_rles.append(rle_mask)

    np.savez_compressed(saved_data_path, 
                        image_paths=image_paths, 
                        mask_rles=mask_rles)
    print(f"Generated and saved {len(image_paths)} images and {len(mask_rles)} RLE masks.")

train_size = int(0.8 * len(image_paths))

train_image_paths = image_paths[:train_size]
train_mask_rles = mask_rles[:train_size]

val_image_paths = image_paths[train_size:]
val_mask_rles = mask_rles[train_size:]

train_dataset = create_dataset(train_image_paths, train_mask_rles)
val_dataset = create_dataset(val_image_paths, val_mask_rles)

def augment_image(img, mask):

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        rotation_angle = tf.random.uniform((), minval=-0.175, maxval=0.175)  
        img = tf.image.rot90(img, k=tf.cast(tf.round(rotation_angle / 0.5), tf.int32))  # Approximate rotation
        mask = tf.image.rot90(mask, k=tf.cast(tf.round(rotation_angle / 0.5), tf.int32))  

    img = tf.image.random_brightness(img, 0.1)

    mask = tf.cast(mask > 0.5, tf.float32)

    return img, mask

train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

def visualize_random_sample(dataset, batch_size):

    random_batch_index = random.randint(0, len(dataset) - 1)
    random_image_index = random.randint(0, batch_size - 1)

    for img_batch, mask_batch in dataset.skip(random_batch_index).take(1):
        sample_image = img_batch[random_image_index].numpy()
        sample_mask = mask_batch[random_image_index].numpy().squeeze()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask, cmap='gray')
    plt.title("Decoded Mask from RLE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

tf.keras.backend.clear_session()
gc.collect()

print("GPU cache cleared")

class LoggedEarlyStopping(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.stopped_epoch > 0:
            print(f"[EARLY STOPPING] Activated at Epoch {epoch + 1} due to no improvement.")

class LoggedModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.best == logs.get(self.monitor):
            print(f"[CHECKPOINT SAVED] Model improved at Epoch {epoch + 1} with {self.monitor}: {logs[self.monitor]:.4f}")

class LoggedReduceLROnPlateau(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        # Correct learning rate retrieval
        if hasattr(self.model.optimizer, 'lr'):
            current_lr = float(K.get_value(self.model.optimizer.lr))
        else:
            current_lr = 'Unknown'

        print(f"[LEARNING RATE ADJUSTED] Current LR: {current_lr}")

early_stopping = LoggedEarlyStopping(
    monitor='val_iou_score',
    patience=10,
    mode='max',
    restore_best_weights=True
)

checkpoint_callback = LoggedModelCheckpoint(
    './checkpoint/model_checkpoint.keras',
    monitor='val_iou_score',
    save_best_only=True,
    mode='max',
    verbose=1
)

lr_schedule = LoggedReduceLROnPlateau(
    monitor='val_iou_score',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Load the saved model
# model = keras.models.load_model(
#      '/kaggle/working/final_trained_model_1.keras',
#      custom_objects={'iou_score': iou_score}  # Important! Register custom metrics
#  )

# print("Pretrained model loaded successfully!")

with strategy.scope():
    model = unet_model()  # Model creation inside strategy scope
    model.compile(
        optimizer=optimizers.Adam(learning_rate=initial_learning_rate),
        loss=combined_loss,
        metrics=['accuracy', iou_score]
    )


with strategy.scope():
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[checkpoint_callback, lr_schedule] # early_stopping
    )

def get_next_model_filename(base_path='./models/final_trained_model'):
    """Finds the next available model filename with auto-incrementing number."""
    num = 1
    while os.path.exists(f"{base_path}_{num}.keras"):
        num += 1
    return f"{base_path}_{num}.keras"

model_path = get_next_model_filename()
print(f"Saving model as: {model_path}")

model.save(model_path)
print("Model saved successfully!")

results = model.evaluate(val_dataset)
print(f"Evaluation Results: {results}")

def save_rle_to_csv(rle_data, output_csv_path="../submission2.csv"):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'EncodedPixels'])  # Header

        for image_id, rle_mask in rle_data.items():
            writer.writerow([image_id, rle_mask])

    print(f"CSV file saved successfully at: {output_csv_path}")


test_dir = "../dataset/test/test/images/"

def generate_rle_masks(model, test_dir):

    rle_data = {}

    # Iterate through test images
    for img_name in sorted(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256)) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict the mask
        predicted_mask = model.predict(img, verbose=0)[0]

        # Post-process the predicted mask as a duotone (0 or 255)
        binary_mask = (predicted_mask > 0.5).astype(np.uint8)

        # Convert to RLE
        rle_mask = mask2rle(binary_mask)

        # Save to dictionary
        rle_data[img_name.replace('.png', '')] = rle_mask

    print(f"Generated RLE masks for {len(rle_data)} test images.")
    return rle_data

rle_data = generate_rle_masks(model, test_dir)
save_rle_to_csv(rle_data)

# Save as .npz file
# np.savez_compressed('..//rle_test_data.npz', rle_data=rle_data)
# print("RLE data saved successfully as `rle_test_data.npz`")

while True:
    random_image_num = random.randint(int(28100 * 0.8), 28100)
    sample_img_name = f"image_{random_image_num}.png"

    sample_img_path = f"../dataset/train/train/images/{sample_img_name}"
    sample_mask_path = f"../dataset/train/train/masks/{sample_img_name}"
    
    sample_img = cv2.imread(sample_img_path)

    if sample_img is not None:
        break

print(f"Displaying image {sample_img_name}")

sample_img_resized = cv2.resize(sample_img, (256, 256)) / 255.0
sample_img_resized = np.expand_dims(sample_img_resized, axis=0)

predicted_mask = model.predict(sample_img_resized, verbose=0)[0]
predicted_mask = postprocess_mask(predicted_mask)

print("Sample mask path:", sample_mask_path)

if os.path.exists(sample_mask_path):
    actual_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
    actual_mask_resized = cv2.resize(actual_mask, (256, 256))
else:
    actual_mask_resized = np.zeros((256, 256))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(sample_img)
plt.title("Sample Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(actual_mask_resized, cmap='gray')
plt.title("Actual Test Mask" if os.path.exists(sample_mask_path) else "No Actual Mask Available")
plt.axis("off")

plt.tight_layout()
plt.show()
