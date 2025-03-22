import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# image_dir = "/kaggle/input/aaltoescv/train/train/images/"
# mask_dir = "/kaggle/input/aaltoescv/train/train/masks/"
image_dir = "../dataset_unzipped/train/train/images/"
mask_dir = "../dataset_unzipped/train/train/masks/"

print("Current Working Directory:", os.getcwd())


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, label=1, shape=(256, 256)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)


data = []

for mask_filename in sorted(os.listdir(mask_dir)):
    if mask_filename.endswith('.png'):
        image_path = os.path.join(image_dir, mask_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 128).astype(np.uint8)
        rle_mask = mask2rle(binary_mask)
        data.append({
            'image_path': image_path,
            'rle_mask': rle_mask
        })

df = pd.DataFrame(data)

# Display sample image and mask
idx = 0
image_sample = cv2.imread(df.iloc[idx]['image_path'])
mask_sample = cv2.imread(os.path.join(mask_dir, os.path.basename(
    df.iloc[idx]['image_path'])), cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_sample, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(mask_sample, cmap='gray')
plt.title("Binary Mask")

plt.show()
