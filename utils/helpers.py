import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array

# These two lines MUST be here
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
TUMOR_THRESHOLD = 0.50

def load_and_preprocess_image(file, target_size=(224, 224)):
    img = load_img(file, target_size=target_size, color_mode='grayscale')
    img_arr = img_to_array(img).astype("float32") / 255.0
    img_tensor = np.expand_dims(img_arr, axis=0)
    return img_tensor, img_arr[:, :, 0]

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")

def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = Model(model.input, [model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)
