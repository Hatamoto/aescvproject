import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print("CUDA version:", tf_build_info.cuda_version_number)
#print("cuDNN version:", tf_build_info.cudnn_version_number)

from tensorflow import keras
print("TensorFlow is working correctly!")
