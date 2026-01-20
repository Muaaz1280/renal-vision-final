# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
import os
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.helpers import (
    CLASS_NAMES,
    TUMOR_THRESHOLD,
    load_and_preprocess_image,
    get_last_conv_layer_name,
    make_gradcam_heatmap
)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING WITH ERROR HANDLING
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/Renal Vision Final.h5"

try:
    st.info("Loading Phase 1 model...")
    phase1_model = load_model(MODEL_PATH)
    st.success("✅ Phase 1 model loaded successfully!")

    # Phase 2 model architecture – EXPLICIT NAMES to prevent collision
    st.info("Building Phase 2 model architecture...")
    inputs = Input(shape=(224, 224, 1), name="input_layer")

    x = Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_1")(inputs)
    x = MaxPooling2D(2,2, name="maxpool_1")(x)

    x = Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_2")(x)
    x = MaxPooling2D(2,2, name="maxpool_2")(x)

    x = Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_3")(x)
    x = MaxPooling2D(2,2, name="maxpool_3")(x)

    x = Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_4")(x)
    x = MaxPooling2D(2,2, name="maxpool_4")(x)

    x = Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_5")(x)
    x = MaxPooling2D(2,2, name="maxpool_5")(x)

    x = Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name="conv2d_6")(x)
    x = MaxPooling2D(2,2, name="maxpool_6")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(128, activation='relu', name="dense_1")(x)
    x = Dropout(0.4, name="dropout")(x)
    outputs = Dense(4, activation='softmax', name="output")(x)

    phase2_model = tf.keras.Model(inputs, outputs, name="phase2_model")
    
    st.info("Loading Phase 2 weights...")
    phase2_model.load_weights(MODEL_PATH)
    st.success("✅ Phase 2 model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.exception(e)
    st.stop()  # Stop execution if models fail to load

# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Renal Vision – Kidney Disease Classification + Grad-CAM")
st.write("Upload a grayscale kidney scan image (JPG, JPEG, PNG)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.info("Preprocessing image...")
        img_tensor, img_gray = load_and_preprocess_image(uploaded_file)

        st.image(img_gray, caption="Uploaded Image", use_column_width=True, clamp=True, channels="GRAY")

        # Phase 1 Prediction
        st.subheader("Phase 1: Classification")
        preds = phase1_model.predict(img_tensor)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = preds[pred_idx]

        st.write(f"**Predicted: {pred_class}** ({pred_conf:.1%} confidence)")
        st.write("Probabilities:")
        for cls, prob in zip(CLASS_NAMES, preds):
            st.write(f"- {cls}: {prob:.1%}")

        # Phase 2: Grad-CAM only for Tumor with good confidence
        if pred_class == "Tumor" and pred_conf >= TUMOR_THRESHOLD:
            st.subheader("Phase 2: Tumor Localization (Grad-CAM)")
            with st.spinner("Computing Grad-CAM..."):
                last_conv = get_last_conv_layer_name(phase2_model)
                heatmap, _ = make_gradcam_heatmap(img_tensor, phase2_model, last_conv)

                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                original_uint8 = (img_gray * 255).astype(np.uint8)
                overlay = cv2.addWeighted(
                    cv2.cvtColor(original_uint8, cv2.COLOR_GRAY2BGR), 0.6,
                    heatmap_color, 0.4, 0
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_uint8, caption="Original", channels="GRAY", use_column_width=True)
                with col2:
                    st.image(heatmap_color, caption="Grad-CAM Heatmap", use_column_width=True)
                with col3:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_column_width=True)
        else:
            st.info(f"Phase 2 skipped ({pred_class} detected)")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Renal Vision – Built with Streamlit & TensorFlow | Deployed on Streamlit Cloud")