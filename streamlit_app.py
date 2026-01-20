import streamlit as st
import numpy as np
import cv2
import os
import traceback

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from utils.helpers import CLASS_NAMES, TUMOR_THRESHOLD, load_and_preprocess_image, get_last_conv_layer_name, make_gradcam_heatmap
    st.success("✅ TensorFlow imported successfully!")
except Exception as e:
    st.error(f"Failed to import TensorFlow: {str(e)}")
    st.exception(e)
    st.stop()  # Stop execution if imports fail

# ── Model Loading with Error Handling ────────────────────────────────────────
MODEL_PATH = "models/Renal Vision Final.h5"

try:
    st.info("Loading Phase 1 model...")
    phase1_model = load_model(MODEL_PATH)
    st.success("✅ Phase 1 model loaded!")

    # Phase 2 architecture (only if Phase 1 succeeds)
    st.info("Building & loading Phase 2 model...")
    inputs = Input(shape=(224, 224, 1))
    x = Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(128, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(4, activation='softmax')(x)

    phase2_model = tf.keras.Model(inputs, outputs)
    phase2_model.load_weights(MODEL_PATH)
    st.success("✅ Phase 2 model loaded!")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.exception(e)
    st.stop()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Renal Vision – Kidney Disease Classification + Grad-CAM")
st.write("Upload a grayscale kidney scan image (JPG/PNG)")

uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img_tensor, img_gray = load_and_preprocess_image(uploaded_file)
        st.image(img_gray, caption="Uploaded Image", use_column_width=True, clamp=True, channels="GRAY")

        # Phase 1
        preds = phase1_model.predict(img_tensor)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = preds[pred_idx]

        st.subheader("Prediction")
        st.write(f"**{pred_class}** ({pred_conf:.1%} confidence)")

        # Phase 2
        if pred_class == "Tumor" and pred_conf >= TUMOR_THRESHOLD:
            st.subheader("Tumor localization (Grad-CAM)")
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
            with col1: st.image(original_uint8, caption="Original", channels="GRAY")
            with col2: st.image(heatmap_color, caption="Heatmap")
            with col3: st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay")
        else:
            st.info(f"Phase 2 skipped ({pred_class} detected)")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.exception(e)