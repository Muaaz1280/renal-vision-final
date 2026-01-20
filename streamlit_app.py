import streamlit as st
# Comment these out for test:
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# ... all model loading and prediction code ...

st.title("Renal Vision – Test Mode")
st.write("If you see this, basic Streamlit works. Model loading is the problem.")

uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded (test)")